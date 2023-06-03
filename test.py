import logging
import time

import torch

from utils.eval_metrics import evaluate, evaluate_with_clothes
from utils.evaluate import extract_vid_feature


def test(config, model, queryloader, galleryloader, dataset):
    logger = logging.getLogger('reid.test')
    since = time.time()
    model.eval()

    query_features, query_pids, query_camids, query_clothes_ids = extract_vid_feature(
        model=model,
        dataloader=queryloader,
        vid2clip_index=dataset.query_vid2clip_index,
        data_length=len(dataset.recombined_query),
        logger=logger)
    gallery_features, gallery_pids, gallery_camids, gallery_clothes_ids = extract_vid_feature(
        model,
        galleryloader,
        dataset.gallery_vid2clip_index,
        len(dataset.recombined_gallery),
        logger=logger)

    torch.cuda.empty_cache()
    time_elapsed = time.time() - since

    logger.info("Extracted features for query set, obtained {} matrix".format(
        query_features.shape))
    logger.info(
        "Extracted features for gallery set, obtained {} matrix".format(
            gallery_features.shape))
    logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    since = time.time()
    m, n = query_features.size(0), gallery_features.size(0)
    distance_matrix = torch.zeros((m, n))
    query_features, gallery_features = query_features.cuda(
    ), gallery_features.cuda()
    # Cosine similarity
    for i in range(m):
        distance_matrix[i] = (
            -torch.mm(query_features[i:i + 1], gallery_features.t())).cpu()
    distance_matrix = distance_matrix.numpy()
    query_pids, query_camids, query_clothes_ids = query_pids.numpy(
    ), query_camids.numpy(), query_clothes_ids.numpy()
    gallery_pids, gallery_camids, gallery_clothes_ids = gallery_pids.numpy(
    ), gallery_camids.numpy(), gallery_clothes_ids.numpy()
    time_elapsed = time.time() - since
    logger.info('Distance computing in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    since = time.time()
    logger.info("Computing CMC and mAP")
    cmc, mAP = evaluate(distance_matrix, query_pids, gallery_pids,
                        query_camids, gallery_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")
    time_elapsed = time.time() - since
    logger.info('Using {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                               time_elapsed % 60))

    logger.info("Computing CMC and mAP only for the same clothes setting")
    cmc, mAP = evaluate_with_clothes(distance_matrix,
                                     query_pids,
                                     gallery_pids,
                                     query_camids,
                                     gallery_camids,
                                     query_clothes_ids,
                                     gallery_clothes_ids,
                                     mode='SC')
    logger.info("Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    logger.info("Computing CMC and mAP only for clothes-changing")
    cmc, mAP = evaluate_with_clothes(distance_matrix,
                                     query_pids,
                                     gallery_pids,
                                     query_camids,
                                     gallery_camids,
                                     query_clothes_ids,
                                     gallery_clothes_ids,
                                     mode='CC')
    logger.info("Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    return cmc[0]
