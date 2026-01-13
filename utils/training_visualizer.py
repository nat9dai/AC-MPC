"""Training visualization utility for AC-DMPC environments."""

from __future__ import annotations

import threading
import time
from typing import Optional, Dict, Any
import numpy as np
import torch

# Set matplotlib backend for interactive display
try:
    import matplotlib
    # Try to use interactive backend
    try:
        matplotlib.use("TkAgg")
    except Exception:
        try:
            matplotlib.use("Qt5Agg")
        except Exception:
            pass  # Use default backend
except ImportError:
    pass


class TrainingVisualizer:
    """Visualizes a single environment during training.
    
    This class runs visualization in a separate thread to avoid blocking training.
    """
    
    def __init__(
        self,
        env,
        agent,
        *,
        history_window: int,
        device: str = "cpu",
        update_interval: float = 0.05,  # Update every 50ms
        max_steps: int = 1000,
    ):
        """Initialize the visualizer.
        
        Args:
            env: Gymnasium environment to visualize
            agent: ActorCriticAgent to use for actions
            history_window: History window size for agent
            device: Device for agent
            update_interval: Time between visualization updates (seconds)
            max_steps: Maximum steps per episode
        """
        self.env = env
        self.agent = agent
        self.history_window = history_window
        self.device = device
        self.update_interval = update_interval
        self.max_steps = max_steps
        
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._training_agent = None  # Reference to training agent for weight updates
        
        # Set render mode if available
        if hasattr(env, 'metadata') and 'human' in env.metadata.get('render_modes', []):
            self.env.render_mode = 'human'
    
    def set_training_agent(self, training_agent):
        """Set reference to training agent for periodic weight updates."""
        self._training_agent = training_agent
    
    def update_weights(self):
        """Update visualizer agent weights from training agent."""
        if self._training_agent is not None:
            with self._lock:
                try:
                    # Filter out incompatible parameters (x_ref, u_ref) that may have different batch sizes
                    training_state = self._training_agent.state_dict()
                    filtered_state = {
                        key: tensor
                        for key, tensor in training_state.items()
                        if ".cost_module.x_ref" not in key and ".cost_module.u_ref" not in key
                    }
                    self.agent.load_state_dict(filtered_state, strict=False)
                except Exception:
                    pass  # Ignore errors during weight update
    
    def start(self, seed: Optional[int] = None):
        """Start visualization in a separate thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._visualize_loop, args=(seed,), daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop visualization."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        # Close environment safely, handling thread issues
        if hasattr(self.env, 'close'):
            try:
                self.env.close()
            except RuntimeError:
                # Ignore "main thread is not in main loop" errors
                # The figure will be cleaned up when the process exits
                pass
            except Exception:
                # Ignore other errors during cleanup
                pass
    
    def _visualize_loop(self, seed: Optional[int] = None):
        """Main visualization loop running in separate thread."""
        try:
            # Set agent to eval mode
            self.agent.eval()
            
            obs, info = self.env.reset(seed=seed)
            state = obs[:self.agent.actor.state_dim].astype(np.float32)
            
            # Initialize agent state
            history = torch.zeros(self.history_window, self.agent.actor.state_dim, device=self.device)
            history[-1] = torch.from_numpy(state).to(self.device)
            memories = self.agent.init_state(batch_size=1)
            warm_start = None
            
            step = 0
            update_counter = 0
            while self.running and step < self.max_steps:
                # Periodically update agent weights from training agent
                update_counter += 1
                if update_counter % 10 == 0:  # Update every 10 steps
                    self.update_weights()
                
                # Get action from agent
                history_batch = history.unsqueeze(0)
                state_tensor = torch.from_numpy(state).to(self.device).unsqueeze(0)
                waypoint = info.get("target_waypoint")
                if waypoint is not None:
                    waypoint_tensor = torch.from_numpy(waypoint).to(self.device).view(1, 1, -1)
                else:
                    waypoint_tensor = None
                
                with torch.no_grad():
                    # Ensure batch size is 1 for single environment
                    action_tensor, memories, plan = self.agent.act(
                        history_batch,  # Shape: [1, history_window, state_dim]
                        state=state_tensor,  # Shape: [1, state_dim]
                        memories=memories,
                        waypoint_seq=waypoint_tensor,  # Shape: [1, 1, waypoint_dim]
                        warm_start=warm_start,
                        return_plan=True,
                    )
                
                # Extract action for single environment
                if action_tensor.dim() > 1:
                    action = action_tensor.squeeze(0).detach().cpu().numpy()
                else:
                    action = action_tensor.detach().cpu().numpy()
                
                # Update warm_start if plan is available
                if plan is not None:
                    # plan[1] is the action plan, ensure it has correct batch dimension
                    warm_start_plan = plan[1].detach()
                    # If warm_start_plan has batch dimension > 1, take first element
                    if warm_start_plan.dim() == 3 and warm_start_plan.shape[0] > 1:
                        warm_start = warm_start_plan[0:1]  # Keep batch dimension as 1
                    elif warm_start_plan.dim() == 2:
                        # Add batch dimension if missing
                        warm_start = warm_start_plan.unsqueeze(0)
                    else:
                        warm_start = warm_start_plan
                
                # Step environment
                next_obs, reward, terminated, truncated, next_info = self.env.step(action)
                done = terminated or truncated
                
                # Render
                if hasattr(self.env, 'render'):
                    self.env.render(mode='human')
                
                # Update state
                next_state = next_obs[:self.agent.actor.state_dim].astype(np.float32)
                history = torch.roll(history, shifts=-1, dims=0)
                history[-1] = torch.from_numpy(next_state).to(self.device)
                state = next_state
                info = next_info
                step += 1
                
                if done:
                    obs, info = self.env.reset()
                    state = obs[:self.agent.actor.state_dim].astype(np.float32)
                    history = torch.zeros(self.history_window, self.agent.actor.state_dim, device=self.device)
                    history[-1] = torch.from_numpy(state).to(self.device)
                    memories = self.agent.init_state(batch_size=1)
                    warm_start = None
                    step = 0
                
                time.sleep(self.update_interval)
        except Exception as e:
            print(f"Visualization error: {e}")
        finally:
            self.running = False


def create_training_visualizer(
    env_factory,
    agent,
    *,
    history_window: int,
    device: str = "cpu",
    env_id: int = 0,
    **env_kwargs
) -> TrainingVisualizer:
    """Create a visualizer for training.
    
    Args:
        env_factory: Function that creates an environment
        agent: ActorCriticAgent
        history_window: History window size
        device: Device for agent
        env_id: Environment ID
        **env_kwargs: Additional environment arguments
    
    Returns:
        TrainingVisualizer instance
    """
    env = env_factory(env_id=env_id, **env_kwargs)
    return TrainingVisualizer(
        env=env,
        agent=agent,
        history_window=history_window,
        device=device,
    )

