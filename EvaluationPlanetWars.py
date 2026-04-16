# Import PPO (trained RL model) and the custom Planet Wars environment
from stable_baselines3 import PPO
from PlanetWarsGymnasium import PlanetWarsEnv


# Function to evaluate the trained model against a specific opponent
def evaluate_against_opponent(model_path, opponent_mode, num_episodes=50):
    
    # Create environment and load trained PPO model
    env = PlanetWarsEnv()
    model = PPO.load(model_path)

    # If evaluating self-play, assign the same trained model to the enemy
    if opponent_mode == "self_play":
        env.game.self_play_model = PPO.load(model_path)

    # Initialize tracking variables
    wins = 0
    losses = 0
    draws = 0
    total_reward = 0.0
    total_steps = 0

    # Run multiple evaluation episodes
    for ep in range(num_episodes):
        
        # Reset environment at start of each episode
        obs, _ = env.reset()
        
        # Set opponent type (basic / heuristic / self_play)
        env.game.opponent_mode = opponent_mode

        done = False
        truncated = False
        ep_reward = 0.0
        ep_steps = 0

        # Run one full episode
        while not done and not truncated:
            
            # Model selects action based on current observation
            action, _ = model.predict(obs, deterministic=True)
            
            # Environment processes action and returns new state + reward
            obs, reward, done, truncated, _ = env.step(action)
            
            # Accumulate reward and step count
            ep_reward += reward
            ep_steps += 1

        # Track total metrics
        total_reward += ep_reward
        total_steps += ep_steps

        # Determine outcome of the episode
        if truncated and not done:
            # Episode ended due to time limit
            draws += 1
        elif env.game.winner == "Player":
            wins += 1
        elif env.game.winner == "Enemy":
            losses += 1
        else:
            # Covers edge cases (e.g., draw conditions)
            draws += 1

    # Print summary of evaluation results
    print(f"\n=== Results vs {opponent_mode.upper()} ===")
    print(f"Episodes: {num_episodes}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Draws/Timeouts: {draws}")
    print(f"Win Rate: {wins / num_episodes:.2%}")
    print(f"Loss Rate: {losses / num_episodes:.2%}")
    print(f"Draw Rate: {draws / num_episodes:.2%}")
    print(f"Average Reward: {total_reward / num_episodes:.2f}")
    print(f"Average Steps: {total_steps / num_episodes:.2f}")


# Entry point of the script
if __name__ == "__main__":
    
    # Path to the trained model file
    model_path = "planet_wars_final"

    # Evaluate against different opponent types
    evaluate_against_opponent(model_path, "basic", num_episodes=50)
    evaluate_against_opponent(model_path, "heuristic", num_episodes=50)
    evaluate_against_opponent(model_path, "self_play", num_episodes=50)