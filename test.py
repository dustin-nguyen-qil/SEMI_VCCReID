import logging
import time
import os.path as osp
import torch
from datasets.dataset_loader import build_testloader
from baseline import Inference
from utils.eval_metrics import evaluate, evaluate_with_clothes
from utils.evaluate import extract_vid_feature
from config import CONFIG 
from utils.utils import build_model_name
import numpy as np
import matplotlib.pyplot as plt

model_name = build_model_name()

logging.basicConfig(filename=f"work_space/loggers/test_{model_name}.txt",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logging.info("Testing latest trained ReID model")

def test(model, queryloader, galleryloader, query, gallery):
    logger = logging.getLogger('reid.test')
    since = time.time()
    model.cuda()
    model.eval()
    print('======== Extracting query features ========')
    query_features, query_pids, query_camids, query_clothes_ids = extract_vid_feature(
        model=model,
        dataloader=queryloader,
        vid2clip_index=query.vid2clip_index,
        data_length=len(query.dataset),
        logger=logger)
    print('======== Extracting gallery features ========')
    gallery_features, gallery_pids, gallery_camids, gallery_clothes_ids = extract_vid_feature(
        model,
        galleryloader,
        vid2clip_index=gallery.vid2clip_index,
        data_length=len(gallery.dataset),
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
    query_features, gallery_features = query_features.cuda(), gallery_features.cuda()
    # Cosine similarity
    for i in range(m):
        distance_matrix[i] = (-torch.mm(query_features[i:i + 1], gallery_features.t())).cpu()
    distance_matrix = distance_matrix.numpy()
    query_pids, query_camids, query_clothes_ids = query_pids.numpy(),\
          query_camids.numpy(), query_clothes_ids.numpy()
    gallery_pids, gallery_camids, gallery_clothes_ids = gallery_pids.numpy(),\
          gallery_camids.numpy(), gallery_clothes_ids.numpy()
    time_elapsed = time.time() - since
    logger.info('Distance computing in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    since = time.time()
    logger.info("Computing CMC and mAP")
    standard_cmc, standard_mAP = evaluate(distance_matrix, query_pids, gallery_pids,
                        query_camids, gallery_camids)
    logger.info("Standard Setting Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(
            standard_cmc[0], standard_cmc[4], standard_cmc[9], standard_cmc[19], standard_mAP))
    logger.info("-----------------------------------------------------------")
    time_elapsed = time.time() - since
    logger.info('Using {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                               time_elapsed % 60))

    logger.info("Computing CMC and mAP only for the same clothes setting")
    sc_cmc, sc_mAP = evaluate_with_clothes(distance_matrix,
                                     query_pids,
                                     gallery_pids,
                                     query_camids,
                                     gallery_camids,
                                     query_clothes_ids,
                                     gallery_clothes_ids,
                                     mode='SC')
    logger.info("Same Clothes Setting Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(
            sc_cmc[0], sc_cmc[4], sc_cmc[9], sc_cmc[19], sc_mAP))
    logger.info("-----------------------------------------------------------")

    logger.info("Computing CMC and mAP only for clothes-changing")
    cc_cmc, cc_mAP = evaluate_with_clothes(distance_matrix,
                                     query_pids,
                                     gallery_pids,
                                     query_camids,
                                     gallery_camids,
                                     query_clothes_ids,
                                     gallery_clothes_ids,
                                     mode='CC')
    logger.info("Cloth-changing Setting Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(
            cc_cmc[0], cc_cmc[4], cc_cmc[9], cc_cmc[19], cc_mAP))
    logger.info("-----------------------------------------------------------")

    return (standard_cmc, standard_mAP, sc_cmc, sc_mAP, cc_cmc, cc_mAP)

"""
    Testing
"""

state_dict_path = osp.join(CONFIG.METADATA.SAVE_PATH, model_name)
# state_dict_path = "work_space/save/vccr_150_16_0.0005_shape_sampler_dense.pth"
appearance_model = Inference(CONFIG)
appearance_model.load_state_dict(torch.load(state_dict_path), strict=False)
queryloader, galleryloader, query, gallery = build_testloader()

(standard_cmc, standard_mAP, sc_cmc, sc_mAP, cc_cmc, cc_mAP) = \
    test(appearance_model, queryloader, galleryloader, query, gallery)

print()
print("==============================")

standard_results = f"Standard | R-1: {standard_cmc[0]:.2f} | R-4: {standard_cmc[4]:.2f} | R-10: {standard_cmc[9]:.2f} | mAP: {standard_mAP:.2f}"
sc_results = f"Same Clothes | R-1: {sc_cmc[0]:.2f} | R-4: {sc_cmc[4]:.2f} | R-10: {sc_cmc[9]:.2f} | mAP: {sc_mAP:.2f}"
cc_results = f"Cloth-changing | R-1: {cc_cmc[0]:.2f} | R-4: {cc_cmc[4]:.2f} | R-10: {cc_cmc[9]:.2f} | mAP: {standard_mAP:.2f}"
# Calculate the rank values for the x-axis
ranks = np.arange(1, len(standard_cmc)+1)
ranks = np.arange(1, 41)

# # Plot the CMC curve 
plt.plot(ranks, sc_cmc[:40], '-o', label=sc_results)
plt.plot(ranks, standard_cmc[:40], '-o', label=standard_results)
plt.plot(ranks, cc_cmc[:40], '-x', label=cc_results)

plt.xlabel('Rank')
plt.ylabel('Identification Rate')
plt.title(model_name)
plt.grid(False)
# Save the plot to an output folder
path = f"work_space/output/{model_name[:-4]}.png"
plt.legend()
plt.savefig(path)