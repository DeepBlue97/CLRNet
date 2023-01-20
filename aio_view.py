def view(predictions, img_metas):
    img_metas = [item for img_meta in img_metas.data for item in img_meta]
    for lanes, img_meta in zip(predictions, img_metas):
        img_name = img_meta['img_name']
        # img = cv2.imread(osp.join(self.data_root, img_name))
        # out_file = osp.join(self.cfg.work_dir, 'visualization',
        #                     img_name.replace('/', '_'))
        lanes = [lane.to_array(self.cfg) for lane in lanes]
        # imshow_lanes(
        #     img, 
        #     lanes, 
        #     show=True,
        #     # out_file=out_file
        # )

