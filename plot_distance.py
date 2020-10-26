import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from arguments import get_args
import os

if __name__ == "__main__":

    args = get_args()
    eval_file_path = args.save_dir + args.env_name + '_distance_based_goal_generation_buffer10epochs' + '/seed_' + str(args.seed) + '/eval_avg_final_distance.npy'
    eval_file_1_path = args.save_dir + args.env_name + '_originalHER_buffer10epochs' + '/seed_' + str(args.seed) + '/eval_avg_final_distance.npy'
    eval_file_2_path = args.save_dir + args.env_name + '_originalDDPG_buffer10epochs' + '/seed_' + str(args.seed) + '/eval_avg_final_distance.npy'
    eval_file_3_path = args.save_dir + args.env_name + '_distance_based_goal_generation_withoutHER_buffer10epochs' + '/seed_' + str(args.seed) + '/eval_avg_final_distance.npy'

    # eval_file_path = args.save_dir + args.env_name + '_distance_based_goal_generation_buffer10epochs' + '/seed_' + str(args.seed) + '/eval_success_rates.npy'
    # eval_file_1_path = args.save_dir + args.env_name + '_originalHER_buffer10epochs' + '/seed_' + str(args.seed) + '/eval_success_rates.npy'
    # eval_file_2_path = args.save_dir + args.env_name + '_originalDDPG_buffer10epochs' + '/seed_' + str(args.seed) + '/eval_success_rates.npy'

    if not os.path.isfile(eval_file_path):
        print("Result file do not exist!")
    else:
        length = 100
        data = np.load(eval_file_path)[:length]
        data1 = np.load(eval_file_1_path)[:length]
        data2 = np.load(eval_file_2_path)[:length]
        data3 = np.load(eval_file_3_path)[:length]
        print(data.shape)
        print(data1.shape)
        print(data2.shape)
        print(data3.shape)
        #
        # data_raw = np.load(eval_file_path)[:length]
        # data1_raw = np.load(eval_file_1_path)[:length]
        # data2_raw = np.load(eval_file_2_path)[:length]
        # data3_raw = np.load(eval_file_3_path)[:length]
        # data = []
        # data1 = []
        # data2 = []
        # data3 = []
        # for i in range(len(data3_raw)):
        #     if i%10 == 0:
        #         data.append(data_raw[i].copy())
        #         data1.append(data1_raw[i].copy())
        #         data2.append(data2_raw[i].copy())
        #         data3.append(data3_raw[i].copy())
        # data = np.array(data)
        # data1 = np.array(data1)
        # data2 = np.array(data2)
        # data3 = np.array(data3)
        # np.save(args.save_dir + args.env_name + '_distance_based_goal_generation_buffer10epochs' + '/seed_' + str(args.seed) + '/eval_avg_final_distance_10cycles.npy', data)
        # np.save(args.save_dir + args.env_name + '_originalHER_buffer10epochs' + '/seed_' + str(args.seed) + '/eval_avg_final_distance_10cycles.npy', data1)
        # np.save(args.save_dir + args.env_name + '_originalDDPG_buffer10epochs' + '/seed_' + str(args.seed) + '/eval_avg_final_distance_10cycles.npy', data2)
        # np.save(args.save_dir + args.env_name + '_distance_based_goal_generation_withoutHER_buffer10epochs' + '/seed_' + str(args.seed) + '/eval_avg_final_distance_10cycles.npy', data3)

        # data_epoch = data_epoch[:20]
        # print(data_epoch)
        # print(data_epoch.shape)
        # print(data.shape)
        x = np.linspace(0, len(data), len(data))
        x1 = np.linspace(0, len(data1), len(data1))
        x2 = np.linspace(0, len(data2), len(data2))
        x3 = np.linspace(0, len(data3), len(data3))
        # x_fix = np.linspace(0, data_len, data_len)

        mpl.style.use('ggplot')
        fig = plt.figure(1)
        fig.patch.set_facecolor('white')
        plt.xlabel('Episodes', fontsize=16)
        # plt.ylabel('Distance between final ag and g', fontsize=16)
        # plt.ylim(0.0,5.0)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.minorticks_on()
        plt.grid(which='major',linestyle='-', linewidth='0.5', color='white')
        plt.grid(which='minor',linestyle='-', linewidth='0.5', color='white')
        # plt.title(args.env_name, fontsize=20)
        # plt.title('FetchPush-v1', fontsize=20)

        tick = 2000
        plt.plot(x*tick, data, color='blue', linewidth=2, label='DDPG+HER+PGG')
        plt.plot(x1*tick, data1, color='red', linewidth=2, label='DDPG+HER')
        plt.plot(x2*tick, data2, color='green', linewidth=2, label='DDPG')
        plt.plot(x3*tick, data3, color='orange', linewidth=2, label='DDPG+PGG')


        plt.legend(loc='upper left')

        plt.show()
