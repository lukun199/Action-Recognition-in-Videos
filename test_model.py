import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import torchvision

from encoder import CNN_fc_EmbedEncoder, Res3d
from decoder import DecoderRNN
import torch, cv2

def rcnn_detector(frames):

    detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    detector.to(device)
    detector.eval()

    lib_pth = []

    with torch.no_grad():
        for each_frame in frames[:3]:

            # ToTensor()
            each_frame = torch.from_numpy(each_frame / 255.).permute(2, 0, 1).float().to(device)
            bboxes = detector([each_frame])[0]

            # select the biggest bbox.
            condition = np.array(bboxes['scores'].cpu() > 0.7) & np.array(bboxes['labels'].cpu() == 1)
            human_idx = np.argwhere(condition == True).flatten()

            if human_idx.size > 0:
                tuple_positions = bboxes['boxes'].cpu().numpy().take(human_idx, axis=0)
                tuple_pos_four = list(zip(*tuple_positions))
                ptf = list(map(lambda x: int(min(x)), tuple_pos_four[:2])) + \
                      list(map(lambda x: int(max(x)), tuple_pos_four[2:]))
                lib_pth.append(ptf)
                # print(ptf)
            else:
                lib_pth.append([0, 0, 160, 160])

        each_coordinate = list(zip(*lib_pth))
        each_coordinate = list(map(lambda x: int(sum(x) / len(x)), each_coordinate))

    return each_coordinate


if __name__ == '__main__':

    dict = {0: 'push', 1: 'pullup', 2: 'talk', 3: 'climb', 4: 'shoot_ball', 5: 'run', 6: 'sword', 7: 'fencing',
            8: 'flic_flac', 9: 'golf', 10: 'hug', 11: 'shoot_bow', 12: 'somersault', 13: 'sit', 14: 'kick_ball',
            15: 'stand', 16: 'clap', 17: 'laugh', 18: 'brush_hair', 19: 'pick', 20: 'smoke', 21: 'kick',
            22: 'sword_exercise', 23: 'chew', 24: 'shake_hands', 25: 'shoot_gun', 26: 'ride_horse', 27: 'catch',
            28: 'climb_stairs', 29: 'wave', 30: 'handstand', 31: 'fall_floor', 32: 'cartwheel', 33: 'draw_sword',
            34: 'turn', 35: 'dive', 36: 'hit', 37: 'eat', 38: 'kiss', 39: 'jump', 40: 'drink', 41: 'walk',
            42: 'swing_baseball', 43: 'punch', 44: 'throw', 45: 'pushup', 46: 'pour', 47: 'dribble', 48: 'ride_bike',
            49: 'situp'}

    abnormal_length = 0

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # use CPU or GPU
    three_d_encoder = Res3d().to(device).eval()
    embed_encoder = CNN_fc_EmbedEncoder().to(device).eval()
    rnn_decoder = DecoderRNN().to(device).eval()

    embed_encoder.load_state_dict(torch.load('./ckpt/cnn_encoder_epoch201.pth'))
    rnn_decoder.load_state_dict(torch.load('./ckpt/rnn_decoder_epoch201.pth'))

    print('---state dict loaded----')

    video_dir = '../dataset/val_data'

    # get videos
    with torch.no_grad():
        lib_vidos = os.listdir(video_dir)
        lib_vidos.sort(key = lambda x: int(x[:-4]))
        for video_name in lib_vidos:

            assert '.avi' in video_name
            video_file_path = os.path.join(video_dir, video_name)
            video_cap = cv2.VideoCapture(video_file_path)
            total_frams = video_cap.get(7)
            if total_frams > (16 + 8):
                selected_frames = np.linspace(4, total_frams - 4, 16).astype(int)
            else:
                abnormal_length += 1
                print('------------------ABNORMAL---------------', video_file_path + '\n')
                with open('result.txt', 'a') as f:
                    f.write('------------------ABNORMAL---------------' + video_file_path + '\n')
                continue

            list_thisFrame = list()
            for idx_frame in selected_frames:
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, idx_frame)
                is_done, frame_this = video_cap.read()
                assert is_done
                frame_this = cv2.cvtColor(frame_this, cv2.COLOR_BGR2RGB)
                frame_this = cv2.resize(frame_this, (160, 160))
                list_thisFrame.append(frame_this[np.newaxis, :])

            input_this_video = np.vstack(list_thisFrame)

            boxes = rcnn_detector(input_this_video)
            boxes = np.array(boxes)[np.newaxis, :]
            input_video = three_d_encoder.preprocess(input_this_video).to(device)
            category = rnn_decoder(embed_encoder(three_d_encoder(input_video).unsqueeze(0), boxes))
            category = torch.max(category,1)[1].cpu().numpy()

            with open('result.txt','a') as f:
                f.write(dict[int(category)] + '\n')





