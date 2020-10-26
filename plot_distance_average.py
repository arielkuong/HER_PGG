import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from arguments import get_args
import os

seed = [123, 307, 528, 614, 831]
# seed = [123, 528, 831]

if __name__ == "__main__":

    args = get_args()
    data_len = args.demo_length

    eval_file_0_path = args.save_dir + args.env_name + '_distance_based_goal_generation_buffer10epochs' + '/seed_' + str(seed[0]) + '/eval_avg_final_distance.npy'
    eval_file_1_path = args.save_dir + args.env_name + '_originalHER_buffer10epochs' + '/seed_' + str(seed[0]) + '/eval_avg_final_distance.npy'
    eval_file_2_path = args.save_dir + args.env_name + '_originalDDPG_buffer10epochs' + '/seed_' + str(seed[0]) + '/eval_avg_final_distance.npy'
    eval_file_3_path = args.save_dir + args.env_name + '_distance_based_goal_generation_withoutHER_buffer10epochs' + '/seed_' + str(seed[0]) + '/eval_avg_final_distance.npy'

    eval_file_0_path_1 = args.save_dir + args.env_name + '_distance_based_goal_generation_buffer10epochs' + '/seed_' + str(seed[1]) + '/eval_avg_final_distance.npy'
    eval_file_1_path_1 = args.save_dir + args.env_name + '_originalHER_buffer10epochs' + '/seed_' + str(seed[1]) + '/eval_avg_final_distance.npy'
    eval_file_2_path_1 = args.save_dir + args.env_name + '_originalDDPG_buffer10epochs' + '/seed_' + str(seed[1]) + '/eval_avg_final_distance.npy'
    eval_file_3_path_1 = args.save_dir + args.env_name + '_distance_based_goal_generation_withoutHER_buffer10epochs' + '/seed_' + str(seed[1]) + '/eval_avg_final_distance.npy'

    eval_file_0_path_2 = args.save_dir + args.env_name + '_distance_based_goal_generation_buffer10epochs' + '/seed_' + str(seed[2]) + '/eval_avg_final_distance.npy'
    eval_file_1_path_2 = args.save_dir + args.env_name + '_originalHER_buffer10epochs' + '/seed_' + str(seed[2]) + '/eval_avg_final_distance.npy'
    eval_file_2_path_2 = args.save_dir + args.env_name + '_originalDDPG_buffer10epochs' + '/seed_' + str(seed[2]) + '/eval_avg_final_distance.npy'
    eval_file_3_path_2 = args.save_dir + args.env_name + '_distance_based_goal_generation_withoutHER_buffer10epochs' + '/seed_' + str(seed[2]) + '/eval_avg_final_distance.npy'

    eval_file_0_path_3 = args.save_dir + args.env_name + '_distance_based_goal_generation_buffer10epochs' + '/seed_' + str(seed[3]) + '/eval_avg_final_distance.npy'
    eval_file_1_path_3 = args.save_dir + args.env_name + '_originalHER_buffer10epochs' + '/seed_' + str(seed[3]) + '/eval_avg_final_distance.npy'
    eval_file_2_path_3 = args.save_dir + args.env_name + '_originalDDPG_buffer10epochs' + '/seed_' + str(seed[3]) + '/eval_avg_final_distance.npy'
    eval_file_3_path_3 = args.save_dir + args.env_name + '_distance_based_goal_generation_withoutHER_buffer10epochs' + '/seed_' + str(seed[3]) + '/eval_avg_final_distance.npy'

    eval_file_0_path_4 = args.save_dir + args.env_name + '_distance_based_goal_generation_buffer10epochs' + '/seed_' + str(seed[4]) + '/eval_avg_final_distance.npy'
    eval_file_1_path_4 = args.save_dir + args.env_name + '_originalHER_buffer10epochs' + '/seed_' + str(seed[4]) + '/eval_avg_final_distance.npy'
    eval_file_2_path_4 = args.save_dir + args.env_name + '_originalDDPG_buffer10epochs' + '/seed_' + str(seed[4]) + '/eval_avg_final_distance.npy'
    eval_file_3_path_4 = args.save_dir + args.env_name + '_distance_based_goal_generation_withoutHER_buffer10epochs' + '/seed_' + str(seed[4]) + '/eval_avg_final_distance.npy'


    if not os.path.isfile(eval_file_0_path):
        print("Result file do not exist!")
    else:
        length = 15
        data00 = np.load(eval_file_0_path)[:length]
        data01 = np.load(eval_file_1_path)[:length]
        data02 = np.load(eval_file_2_path)[:length]
        data03 = np.load(eval_file_3_path)[:length]

        data10 = np.load(eval_file_0_path_1)[:length]
        data11 = np.load(eval_file_1_path_1)[:length]
        data12 = np.load(eval_file_2_path_1)[:length]
        data13 = np.load(eval_file_3_path_1)[:length]

        data20 = np.load(eval_file_0_path_2)[:length]
        data21 = np.load(eval_file_1_path_2)[:length]
        data22 = np.load(eval_file_2_path_2)[:length]
        data23 = np.load(eval_file_3_path_2)[:length]

        data30 = np.load(eval_file_0_path_3)[:length]
        data31 = np.load(eval_file_1_path_3)[:length]
        data32 = np.load(eval_file_2_path_3)[:length]
        data33 = np.load(eval_file_3_path_3)[:length]

        data40 = np.load(eval_file_0_path_4)[:length]
        data41 = np.load(eval_file_1_path_4)[:length]
        data42 = np.load(eval_file_2_path_4)[:length]
        data43 = np.load(eval_file_3_path_4)[:length]

        x = np.linspace(0, len(data00), len(data00))
        x1 = np.linspace(0, len(data01), len(data01))
        x2 = np.linspace(0, len(data02), len(data02))
        x3 = np.linspace(0, len(data03), len(data03))

        data_comb_0 = [data00, data10, data20, data30, data40]
        # data_comb_0 = [data00, data10, data20]
        data_mean_0 = np.mean(data_comb_0, axis=0)
        data_std_0 = np.std(data_comb_0, axis=0)
        data_low_0 = data_mean_0 - data_std_0
        data_high_0 = data_mean_0 + data_std_0

        data_comb_1 = [data01, data11, data21, data31, data41]
        # data_comb_1 = [data01, data11, data21]
        data_mean_1 = np.mean(data_comb_1, axis=0)
        data_std_1 = np.std(data_comb_1, axis=0)
        data_low_1 = data_mean_1 - data_std_1
        data_high_1 = data_mean_1 + data_std_1

        data_comb_2 = [data02, data12, data22, data32, data42]
        # data_comb_2 = [data02, data12, data22]
        data_mean_2 = np.mean(data_comb_2, axis=0)
        data_std_2 = np.std(data_comb_2, axis=0)
        data_low_2 = data_mean_2 - data_std_2
        data_high_2 = data_mean_2 + data_std_2

        data_comb_3 = [data03, data13, data23, data33, data43]
        # data_comb_3 = [data03, data13, data23]
        data_mean_3 = np.mean(data_comb_3, axis=0)
        data_std_3 = np.std(data_comb_3, axis=0)
        data_low_3 = data_mean_3 - data_std_3
        data_high_3 = data_mean_3 + data_std_3



        mpl.style.use('ggplot')
        fig = plt.figure(1)
        fig.patch.set_facecolor('white')
        plt.xlabel('Episodes', fontsize=16)
        plt.ylabel('Median final distance', fontsize=16)
        # plt.ylim(0.0,0.6)
        plt.minorticks_on()
        plt.grid(which='major',linestyle='-', linewidth='0.5', color='white')
        plt.grid(which='minor',linestyle='-', linewidth='0.5', color='white')
        # plt.title(args.env_name, fontsize=20)

        tick = 2000
        plt.plot(x1*tick, data_mean_1, color='red', linewidth=2, label='DDPG+HER')
        plt.fill_between(x1*tick, data_low_1, data_high_1, color='red', alpha=0.1)
        plt.plot(x2*tick, data_mean_2, color='green', linewidth=2, label='DDPG')
        plt.fill_between(x2*tick, data_low_2, data_high_2, color='green', alpha=0.1)
        plt.plot(x3*tick, data_mean_3, color='orange', linewidth=2, label='DDPG+PGG')
        plt.fill_between(x3*tick, data_low_3, data_high_3, color='orange', alpha=0.1)
        plt.plot(x*tick, data_mean_0, color='blue', linewidth=2, label='DDPG+HER+PGG,sparse')
        plt.fill_between(x*tick, data_low_0, data_high_0, color='blue', alpha=0.1)

        plt.legend(loc='upper right')

        plt.show()
