## backed up from model_rec.py
for f in flist:
    hm = Hand_Model(f)
    nail_size_vec = []
    for i in range(5):
        vf = hm.full_cross_model[i]
        min_dist = 9999
        min_idx = 0
        for j in range( NUM_COMBI ):
            dist = cm_set.combi_models[j].geometric_distance_to( vf, i )
            # d1 = distance.euclidean( vh, vm1)
            print( 'Finger {} Model {} - Euclidean distance: {:07.2f} full cross'.format( i, j, dist) )
            if (  dist <= min_dist):
                min_dist = dist
                min_idx = j
        if i == 4 :
            nail_size_vec.append(SIZE_CHART[i][min_idx])
            print("Thumb is classified to combi id {} in {} - size {} ".format( min_idx,
                                                        COMBI_THUMBS[min_idx], SIZE_CHART[4][min_idx]))
        else:
            sz = SIZE_CHART[i][min_idx]
            nail_size_vec.append(sz)
            print( "Finger {} is classified to combi id {} in {} - size {}".format( i,
                                                        min_idx, COMBI_FINGERS[min_idx], sz ))
    nschart = np.array(SIZE_CHART)
    min_dist = 9999
    min_idx = 0
    dist_vec = []
    for i in range(NUM_COMBI):
        combi_size_vec = nschart[:,i]
        dist = distance.euclidean( nail_size_vec, combi_size_vec )
        dist_vec.append(dist)
        if dist < min_dist :
            min_dist = dist
            min_idx = i
    print( "Nail size classification assigns the hand to combi {}".format( min_idx + 1))
    print( "Nails classified to sizes: ", nail_size_vec )
    print()


class DataGenerator:
    def test(self):
        b = 0
        batch_images = []
        batch_image_meta = []
        batch_rpn_match = []
        batch_rpn_bbox = []
        batch_gt_class_ids = []
        batch_gt_boxes = []
        batch_gt_masks = []
        no_augmentation_sources = self.no_augmentation_sources or []
        while True:
            try:
                # Increment index to pick next image. Shuffle if at the start of an epoch.
                self.image_index = (self.image_index + 1) % len(self.image_ids)

                # Get GT bounding boxes and masks for image.
                image_id = self.image_ids[self.image_index]

                # If the image source is not to be augmented pass None as augmentation
                print( "no_augmentation_sources : ", no_augmentation_sources)
                if self.dataset.image_info[image_id]['source'] in no_augmentation_sources:
                    print( "no image augmentation")
                    image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                        load_image_gt( self.dataset, self.config, image_id, augment=self.augment,
                                      augmentation=None,
                                      use_mini_mask=self.config.USE_MINI_MASK)
                else:
                    image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                        load_image_gt( self.dataset, self.config, image_id, augment=self.augment,
                                      augmentation=self.augmentation,
                                      use_mini_mask=self.config.USE_MINI_MASK)

                # Skip images that have no instances. This can happen in cases
                # where we train on a subset of classes and the image doesn't
                # have any of the classes we care about.
                if not np.any(gt_class_ids > 0):
                    continue

                # RPN Targets
                rpn_match, rpn_bbox = build_rpn_targets(image.shape, self.anchors,
                                                        gt_class_ids, gt_boxes, self.config)

                # Mask R-CNN Targets
                if self.random_rois:
                    rpn_rois = generate_random_rois(
                        image.shape, self.random_rois, gt_class_ids, gt_boxes)
                    if self.detection_targets:
                        rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask = \
                            build_detection_targets( rpn_rois, gt_class_ids, gt_boxes, gt_masks, self.config)

                # Init batch arrays
                if b == 0:
                    batch_image_meta = np.zeros(
                        (self.batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                    batch_rpn_match = np.zeros(
                        [self.batch_size, self.anchors.shape[0], 1], dtype=rpn_match.dtype)
                    batch_rpn_bbox = np.zeros(
                        [self.batch_size, self.config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                    batch_images = np.zeros(
                        (self.batch_size,) + image.shape, dtype=np.float32)
                    batch_gt_class_ids = np.zeros(
                        (self.batch_size, self.config.MAX_GT_INSTANCES), dtype=np.int32)
                    batch_gt_boxes = np.zeros(
                        (self.batch_size, self.config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                    batch_gt_masks = np.zeros(
                        (self.batch_size, gt_masks.shape[0], gt_masks.shape[1],
                         self.config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)
                    if self.random_rois:
                        batch_rpn_rois = np.zeros( (self.batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                        if self.detection_targets:
                            batch_rois = np.zeros(
                                (self.batch_size,) + rois.shape, dtype=rois.dtype)
                            batch_mrcnn_class_ids = np.zeros(
                                (self.batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
                            batch_mrcnn_bbox = np.zeros(
                                (self.batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
                            batch_mrcnn_mask = np.zeros(
                                (self.batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)

                # If more instances than fits in the array, sub-sample from them.
                if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
                    ids = np.random.choice(
                        np.arange(gt_boxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
                    gt_class_ids = gt_class_ids[ids]
                    gt_boxes = gt_boxes[ids]
                    gt_masks = gt_masks[:, :, ids]

                # Add to batch
                batch_image_meta[b] = image_meta
                batch_rpn_match[b] = rpn_match[:, np.newaxis]
                batch_rpn_bbox[b] = rpn_bbox
                batch_images[b] = mold_image(image.astype(np.float32), self.config)
                batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
                batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
                batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
                if self.random_rois:
                    batch_rpn_rois[b] = rpn_rois
                    if self.detection_targets:
                        batch_rois[b] = rois
                        batch_mrcnn_class_ids[b] = mrcnn_class_ids
                        batch_mrcnn_bbox[b] = mrcnn_bbox
                        batch_mrcnn_mask[b] = mrcnn_mask
                if b >= self.batch_size:
                    break
                b += 1

            except (GeneratorExit, KeyboardInterrupt):
                raise
            except:
                # Log it and skip the image
                logging.exception("Error processing image {}".format(
                    self.dataset.image_info[image_id]))
                self.error_count += 1
                if self.error_count > 5:
                    raise

        # end of for-loop, batch full
        inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                  batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
        outputs = []

        if self.random_rois:
            inputs.extend([batch_rpn_rois])
            if self.detection_targets:
                inputs.extend([batch_rois])
                inputs.extend([batch_rois])
                # Keras requires that output and targets have the same number of dimensions
                batch_mrcnn_class_ids = np.expand_dims(
                    batch_mrcnn_class_ids, -1)
                outputs.extend(
                    [batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])

        return [ inputs, outputs ]
