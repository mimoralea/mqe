#!/usr/bin/env python
from openrl_ws.utils import make_env, get_args
from mqe.envs.utils import custom_cfg

from openrl.envs.common import make
from openrl.modules.common import PPONet
from openrl.runners.common import PPOAgent

import cv2
import imageio
import numpy as np
import time
import os
from isaacgym import gymapi

def save_video(frames, fps, output_video_path='output_video.mp4'):
    """
    Save frames to a video file.

    Args:
        frames: Either a list of frames or a numpy array of frames
        fps: Frames per second
        output_video_path: Path to save the video
    """
    # Convert frames to numpy array if it's a list
    if isinstance(frames, list):
        # Check if we have any frames
        if not frames:
            print("No frames to save")
            return

        # Get the shape of a single frame to determine dimensions
        sample_frame = frames[0]
        if isinstance(sample_frame, np.ndarray):
            # If frames are already numpy arrays, stack them
            frames_array = np.stack(frames)
        else:
            # Otherwise, convert to numpy arrays and stack
            frames_array = np.stack([np.array(frame) for frame in frames])
    else:
        # Already a numpy array
        frames_array = frames

    # Define the video codec
    codec = cv2.VideoWriter_fourcc(*'mp4v')

    # Get the shape of the frames
    if len(frames_array.shape) == 4:  # [num_frames, height, width, channels] or [num_frames, channels, height, width]
        if frames_array.shape[1] == 3 or frames_array.shape[1] == 4:  # [num_frames, channels, height, width]
            # Transpose to [num_frames, height, width, channels]
            frames_array = np.transpose(frames_array, (0, 2, 3, 1))

        # Ensure we only use the first 3 channels (RGB)
        if frames_array.shape[3] > 3:
            frames_array = frames_array[:, :, :, :3]

        frame_shape = (frames_array.shape[2], frames_array.shape[1])  # (width, height)
    else:
        print(f"Unexpected frame shape: {frames_array.shape}")
        return

    # Create a VideoWriter object
    out = cv2.VideoWriter(output_video_path, codec, fps, frame_shape)

    # Iterate through each frame
    for i in range(len(frames_array)):
        # Convert frame to uint8 if needed
        frame = frames_array[i]
        if frame.dtype != np.uint8:
            # Normalize to 0-255 if not already in that range
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

        # Write the frame to the video file
        out.write(frame)

    # Release the VideoWriter object
    out.release()

    print(f"Video created successfully at: {output_video_path}")

def save_gif(frames, fps, output_gif_path='output_animation.gif'):
    """
    Save frames to a GIF file with maximum quality.

    Args:
        frames: Either a list of frames or a numpy array of frames
        fps: Frames per second
        output_gif_path: Path to save the GIF
    """
    # Convert frames to numpy array if it's a list
    if isinstance(frames, list):
        # Check if we have any frames
        if not frames:
            print("No frames to save")
            return

        # Get the shape of a single frame to determine dimensions
        sample_frame = frames[0]
        if isinstance(sample_frame, np.ndarray):
            # If frames are already numpy arrays, stack them
            frames_array = np.stack(frames)
        else:
            # Otherwise, convert to numpy arrays and stack
            frames_array = np.stack([np.array(frame) for frame in frames])
    else:
        # Already a numpy array
        frames_array = frames

    # Process frames for GIF format
    if len(frames_array.shape) == 4:  # [num_frames, height, width, channels] or [num_frames, channels, height, width]
        if frames_array.shape[1] == 3 or frames_array.shape[1] == 4:  # [num_frames, channels, height, width]
            # Transpose to [num_frames, height, width, channels]
            frames_array = np.transpose(frames_array, (0, 2, 3, 1))

        # Ensure we only use the first 3 channels (RGB)
        if frames_array.shape[3] > 3:
            frames_array = frames_array[:, :, :, :3]
    else:
        print(f"Unexpected frame shape: {frames_array.shape}")
        return

    # Convert frames to uint8 if needed
    if frames_array.dtype != np.uint8:
        # Normalize to 0-255 if not already in that range
        if frames_array.max() <= 1.0:
            frames_array = (frames_array * 255).astype(np.uint8)
        else:
            frames_array = frames_array.astype(np.uint8)

    # Calculate duration for each frame in milliseconds
    # Use a shorter duration for faster/smoother playback (minimum 10ms)
    duration = max(int(1000 / fps), 10)  # Minimum 10ms for browser compatibility

    # Use PIL to create the GIF with proper animation
    import PIL.Image

    # Convert frames to PIL Images with maximum quality
    pil_images = []
    for i in range(len(frames_array)):
        img = PIL.Image.fromarray(frames_array[i])
        # Apply slight sharpening to enhance details
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.2)  # Slight sharpening factor
        pil_images.append(img)

    # Save as GIF with maximum quality settings
    pil_images[0].save(
        output_gif_path,
        save_all=True,
        append_images=pil_images[1:],
        optimize=False,      # Don't optimize to maintain quality
        quality=100,         # Maximum quality
        duration=duration,
        loop=0               # Loop forever
    )

    print(f"Maximum quality GIF created successfully at: {output_gif_path}")

args = get_args()
env, _ = make_env(args, custom_cfg(args))
net = PPONet(env, device="cuda")  # Create neural network.
agent = PPOAgent(net)  # Initialize the agent.

if args.algo == "jrpo" or args.algo == "ppo":
    from openrl.modules.common import PPONet
    from openrl.runners.common import PPOAgent
    net = PPONet(env, cfg=args, device=args.rl_device)
    agent = PPOAgent(net)
else:
    from openrl.modules.common import MATNet
    from openrl.runners.common import MATAgent
    env = MATWrapper(env)
    net = MATNet(env, cfg=args, device=args.rl_device)
    agent = MATAgent(net, use_wandb=args.use_wandb)

if getattr(args, "checkpoint") is not None:
    agent.load(args.checkpoint)

# env.start_recording()
agent.set_env(env)  # The agent requires an interactive environment.

# Track the number of episodes completed
episode_count = 0
max_episodes = args.num_episodes

# Create recordings directory if it doesn't exist
recordings_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recordings")
if not os.path.exists(recordings_dir):
    os.makedirs(recordings_dir)

# Create a task-specific folder
task_name = args.task
task_dir = os.path.join(recordings_dir, task_name)
if not os.path.exists(task_dir):
    os.makedirs(task_dir)

# Create a timestamp folder for this run
timestamp = int(time.time())
timestamp_dir = os.path.join(task_dir, str(timestamp))
if not os.path.exists(timestamp_dir):
    os.makedirs(timestamp_dir)

# Enable video recording if requested
if args.record_video:
    print("Video recording enabled")
    # Try to enable recording in the environment
    if hasattr(env, 'start_recording'):
        env.start_recording()
    # If the environment has a record_now attribute, set it to True
    if hasattr(env, 'record_now'):
        env.record_now = True

obs = env.reset()  # Initialize the environment to obtain initial observations and environmental information.

while episode_count < max_episodes:
    action, _ = agent.act(obs)  # The agent predicts the next action based on environmental observations.
    # The environment takes one step according to the action, obtains the next observation, reward, whether it ends and environmental information.
    obs, r, done, info = env.step(action)

    # Check if episode is done
    if done.any():
        # Save video if recording is enabled
        if args.record_video:
            # Get task name and algorithm name for the filename
            task_name = args.task
            algo_name = args.algo

            # Create paths for video and GIF files with timestamp included in filename
            video_path = os.path.join(timestamp_dir, f"{task_name}_{algo_name}_ep{episode_count+1}_{timestamp}.mp4")
            gif_path = os.path.join(timestamp_dir, f"{task_name}_{algo_name}_ep{episode_count+1}_{timestamp}.gif")

            # Try to get frames from the environment
            frames = None
            if hasattr(env, 'get_complete_frames'):
                frames = env.get_complete_frames()

            if frames and len(frames) > 0:
                print(f"Saving video with {len(frames)} frames")
                # Save the video in MP4 format
                save_video(frames, 30, video_path)

                # Also save as GIF (use a lower fps for GIF to keep file size reasonable)
                # Sample every 2nd frame to reduce GIF size while maintaining high quality
                sampled_frames = frames[::2]

                # Use full resolution for maximum quality
                processed_frames = []
                for frame in sampled_frames:
                    # Convert to numpy array if not already
                    if not isinstance(frame, np.ndarray):
                        frame = np.array(frame)

                    # Keep full resolution
                    if len(frame.shape) == 3:  # [height, width, channels]
                        processed_frames.append(frame)
                    elif len(frame.shape) == 4:  # [channels, height, width]
                        # Just transpose if needed
                        frame_transposed = np.transpose(frame, (1, 2, 0))
                        processed_frames.append(frame_transposed)

                # Save the GIF with the full-resolution frames
                save_gif(processed_frames, 20, gif_path)
            else:
                print("No frames captured for video recording")

                # Create a simple placeholder video with text
                width, height = 640, 480
                fps = 30

                # Create a VideoWriter object for MP4
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

                # Create a frame with text
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(frame, f"Episode {episode_count+1} - {task_name} - {algo_name}",
                           (50, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Write the frame multiple times to create a short video
                for _ in range(fps * 3):  # 3 seconds
                    out.write(frame)

                # Release the VideoWriter
                out.release()

                # Also create a placeholder GIF
                frames_list = [frame] * 30  # 30 frames
                save_gif(frames_list, 10, gif_path)

                print(f"Placeholder videos saved to recordings directory")

        episode_count += 1
        print(f"Episode {episode_count}/{max_episodes} completed")
        if episode_count < max_episodes:
            obs = env.reset()  # Reset for next episode
