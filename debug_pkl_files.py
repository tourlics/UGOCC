import pickle
import mmcv
pkls_files = './data/waymo/kitti_format/waymo_infos_train.pkl'
nusc_pkls_files = './data/nuscenes/bevdetv3-nuscenes-generative_infos_val.pkl'



# todo" 20240627, 我们要做的是什么
# 短期来讲，只要将occ给放入这里就可以
# 唱起来讲，结构中应该包含扩展性的部分    我的工作应该在泛化性和

if __name__=="__main__":
    '''
    understand each label's meaning in waymo's pkl
    '''

    print('all')
    ann_infos = mmcv.load(pkls_files, file_format='pkl')

    print('ann infos type is: ', type(ann_infos), len(ann_infos))
    print(type(ann_infos[0]), ann_infos[0].keys())
    print(ann_infos[0]['pose'].shape)
    print(ann_infos[0]['calib'].keys())
    print(ann_infos[0]['image'].keys())
    print(ann_infos[0]['point_cloud'].keys())
    print(ann_infos[0]['annos'].keys())
    print(len(ann_infos[0]['sweeps']))