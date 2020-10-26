import torch
from models import actor
from arguments import get_args
import gym
import gym_fixed_tasks
import numpy as np

from gym.envs.robotics import rotations
def rotation_goal_distance(goal_a, goal_b, ignore_z_target_rotation = False):
    assert goal_a.shape == goal_b.shape
    assert goal_a.shape[-1] == 7
    d_rot = np.zeros_like(goal_b[..., 0])
    quat_a, quat_b = goal_a[..., 3:], goal_b[..., 3:]

    if ignore_z_target_rotation:
        # Special case: We want to ignore the Z component of the rotation.
        # This code here assumes Euler angles with xyz convention. We first transform
        # to euler, then set the Z component to be equal between the two, and finally
        # transform back into quaternions.
        euler_a = rotations.quat2euler(quat_a)
        euler_b = rotations.quat2euler(quat_b)
        euler_a[2] = euler_b[2]
        quat_a = rotations.euler2quat(euler_a)

    # Subtract quaternions and extract angle between them.
    quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))
    angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1., 1.))
    d_rot = angle_diff

    return d_rot


# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

if __name__ == '__main__':
    args = get_args()
    # create the environment
    env = gym_fixed_tasks.make(args.env_name)
    # get the env param
    observation = env.reset()

    for i in range(args.demo_length):

        # load the model param
        model_path = args.save_dir + args.env_name + '_distance_based_goal_generation_buffer10epochs' + '/seed_' + str(args.seed) + '/model_best.pt'
        # model_path = args.save_dir + args.env_name + '_distance_based_goal_generation_buffer10epochs' + '/seed_' + str(args.seed) + '/model_epoch' + str(args.n_epochs) + '.pt'
        o_mean, o_std, g_mean, g_std, actor_model, critic_model = torch.load(model_path, map_location=lambda storage, loc: storage)
        # get the environment params
        env_params = {'obs': observation['observation'].shape[0],
                      'goal': observation['desired_goal'].shape[0],
                      'action': env.action_space.shape[0],
                      'action_max': env.action_space.high[0],
                      }
        # create the actor network
        actor_network = actor(env_params)
        actor_network.load_state_dict(actor_model)
        actor_network.eval()


        observation = env.reset()
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']
        print(g)
        for t in range(env._max_episode_steps):
            env.render()
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()
            # put actions into the environment
            observation_new, reward, _, info = env.step(action)
            obs = observation_new['observation']
            print(reward)

        ag_final = observation_new['achieved_goal']
        print(ag_final)

        # d_pos,d_rot = env.goal_distance(ag_final, g)
        # print(d_pos)
        # print(d_rot)
        # d_rot1 = rotation_goal_distance(ag_final, g)
        # print(d_rot1)

        print('the episode is: {}, is success: {}'.format(i, info['is_success']))
