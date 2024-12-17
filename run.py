from dataset import *
import evaluation
from fakeflow import FakeFlow
from trainer import Trainer
from optparse import OptionParser
import warnings
warnings.filterwarnings('ignore')


parser = OptionParser()
parser.add_option('--train', action='store_true', default=None)
parser.add_option('--test', action='store_true', default=None)
options = parser.parse_args()[0]


def train_ytvos(model):
    train_set = TrainYTVOS('../DB/YTVOS18', 'train', clip_n=512)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_set = TestDAVIS('../DB/DAVIS', '2016', 'val')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    trainer = Trainer(model, optimizer, train_loader, val_set, save_name='ytvos', save_step=1000, val_step=100)
    trainer.train(4000)


def train_dutsv2_davis(model):
    dutsv2_set = TrainDUTSv2('../DB/DUTSv2', clip_n=384)
    davis_set = TrainDAVIS('../DB/DAVIS', '2016', 'train', clip_n=128)
    train_set = torch.utils.data.ConcatDataset([dutsv2_set, davis_set])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_set = TestDAVIS('../DB/DAVIS', '2016', 'val')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    trainer = Trainer(model, optimizer, train_loader, val_set, save_name='ytvos_dutsv2_davis', save_step=1000, val_step=100)
    trainer.train(2000)


def test_davis(model):
    evaluator = evaluation.Evaluator(TestDAVIS('../DB/DAVIS', '2016', 'val'))
    evaluator.evaluate(model, os.path.join('outputs', 'DAVIS16_val'))


def test_fbms(model):
    test_set = TestFBMS('../DB/FBMS/TestSet')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=4)
    model.cuda()
    ious = []

    # inference
    for vos_data in test_loader:
        imgs = vos_data['imgs'].cuda()
        flows = vos_data['flows'].cuda()
        masks = vos_data['masks'].cuda()
        video_name = vos_data['video_name'][0]
        files = vos_data['files']
        os.makedirs('outputs/FBMS_test/{}'.format(video_name), exist_ok=True)
        vos_out = model(imgs, flows)

        # get iou of each sequence
        iou = 0
        count = 0
        for i in range(masks.size(1)):
            tv.utils.save_image(vos_out['masks'][0, i].float(), 'outputs/FBMS_test/{}/{}'.format(video_name, files[i][0].split('/')[-1]))
            if torch.sum(masks[0, i]) == 0:
                continue
            iou = iou + torch.sum(masks[0, i] * vos_out['masks'][0, i]) / torch.sum((masks[0, i] + vos_out['masks'][0, i]).clamp(0, 1))
            count = count + 1
        print('{} iou: {:.5f}'.format(video_name, iou / count))
        ious.append(iou / count)

    # calculate overall iou
    print('total seqs\' iou: {:.5f}\n'.format(sum(ious) / len(ious)))


def test_ytobj(model):
    test_set = TestYTOBJ('../DB/YTOBJ')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=4)
    model.cuda()
    ious = {'aeroplane': [], 'bird': [], 'boat': [], 'car': [], 'cat': [], 'cow': [], 'dog': [], 'horse': [], 'motorbike': [], 'train': []}
    total_iou = 0
    total_count = 0

    # inference
    for vos_data in test_loader:
        imgs = vos_data['imgs'].cuda()
        flows = vos_data['flows'].cuda()
        masks = vos_data['masks'].cuda()
        class_name = vos_data['class_name'][0]
        video_name = vos_data['video_name'][0]
        files = vos_data['files']
        os.makedirs('outputs/YTOBJ/{}/{}'.format(class_name, video_name), exist_ok=True)
        vos_out = model(imgs, flows)

        # get iou of each sequence
        iou = 0
        count = 0
        for i in range(masks.size(1)):
            tv.utils.save_image(vos_out['masks'][0, i].float(), 'outputs/YTOBJ/{}/{}/{}'.format(class_name, video_name, files[i][0].split('/')[-1]))
            if torch.sum(masks[0, i]) == 0:
                continue
            iou = iou + torch.sum(masks[0, i] * vos_out['masks'][0, i]) / torch.sum((masks[0, i] + vos_out['masks'][0, i]).clamp(0, 1))
            count = count + 1
        if count == 0:
            continue
        print('{}_{} iou: {:.5f}'.format(class_name, video_name, iou / count))
        ious[class_name].append(iou / count)
        total_iou = total_iou + iou / count
        total_count = total_count + 1

    # calculate overall iou
    for class_name in ious.keys():
        print('class: {} seqs\' iou: {:.5f}'.format(class_name, sum(ious[class_name]) / len(ious[class_name])))
    print('total seqs\' iou: {:.5f}\n'.format(total_iou / total_count))


def test_lvid(model):
    test_set = TestLVID('../DB/LVID')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=4)
    model.cuda()
    ious = []

    # inference
    for vos_data in test_loader:
        imgs = vos_data['imgs'].cuda()
        flows = vos_data['flows'].cuda()
        masks = vos_data['masks'].cuda()
        video_name = vos_data['video_name'][0]
        files = vos_data['files']
        os.makedirs('outputs/LVID/{}'.format(video_name), exist_ok=True)
        vos_out = model(imgs, flows)

        # get iou of each sequence
        iou = 0
        count = 0
        for i in range(masks.size(1)):
            tv.utils.save_image(vos_out['masks'][0, i].float(), 'outputs/LVID/{}/{}'.format(video_name, files[i][0].split('/')[-1]))
            if torch.sum(masks[0, i]) == 0:
                continue
            iou = iou + torch.sum(masks[0, i] * vos_out['masks'][0, i]) / torch.sum((masks[0, i] + vos_out['masks'][0, i]).clamp(0, 1))
            count = count + 1
        print('{} iou: {:.5f}'.format(video_name, iou / count))
        ious.append(iou / count)

    # calculate overall iou
    print('total seqs\' iou: {:.5f}\n'.format(sum(ious) / len(ious)))


if __name__ == '__main__':

    # set device
    torch.cuda.set_device(0)

    # define model
    ver = 'mitb2'
    model = FakeFlow(ver).eval()

    # training stage
    if options.train:
        model = torch.nn.DataParallel(model)
        train_ytvos(model)
        train_dutsv2_davis(model)

    # testing stage
    if options.test:
        model.load_state_dict(torch.load('weights/FakeFlow_{}.pth'.format(ver), map_location='cpu'))
        with torch.no_grad():
            test_davis(model)
            test_fbms(model)
            test_ytobj(model)
            test_lvid(model)
