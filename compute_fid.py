from eval_utils import parse_eval_args
from eval_utils import log_fid
from eval_utils import cifar10_to_png
from eval_utils import calculate_fid_given_paths
from eval_utils import set_seed


set_seed()
args = parse_eval_args()
cifar10_to_png(args.total_images)

fid_value = calculate_fid_given_paths(
    paths=[args.dataset1, args.dataset2],
    batch_size=args.batch_size,
    device=args.device,
    total_images=args.total_images,
    dims=2048
)

print(fid_value)

log_fid(fid_value, args.log_path, args.proj_name, args.total_images)

# import torch
# from torchmetrics.image.fid import FrechetInceptionDistance
# from tqdm import tqdm
#
# import os
# import cv2
#
#
# def read_images(dir_path, num_images):
#     gen_list = os.listdir(dir_path)
#     gen_images = []
#
#     for i, image in tqdm(enumerate(gen_list), desc="Loading gen images"):
#         if '.png' not in image:
#             continue
#
#         img = cv2.imread(f'{dir_path}/{image}')
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         # img = img.transpose((2, 0, 1))
#         # img_tensor = torch.tensor(img, dtype=torch.uint8)[None, ...]
#         img = img.transpose((2, 0, 1)) / 255
#         img_tensor = torch.tensor(img, dtype=torch.float64)[None, ...]
#
#         gen_images.append(img_tensor)
#
#         if sum([t.shape[0] for t in gen_images]) >= num_images:
#             break
#
#     gen_images = torch.cat(gen_images)
#     return gen_images
#
# # def read_gen_images(gen_path, num_images):
# #     gen_list = os.listdir(gen_path)
# #     gen_images = []
# #
# #     for i, image in tqdm(enumerate(gen_list), desc="Loading gen images"):
# #         if '.png' not in image:
# #             continue
# #
# #         img = cv2.imread(f'{gen_path}/{image}')
# #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #         img = img.transpose((2, 0, 1))
# #         img_tensor = torch.tensor(img, dtype=torch.uint8)[None, ...]
# #
# #         gen_images.append(img_tensor)
# #
# #         if sum([t.shape[0] for t in gen_images]) >= num_images:
# #             break
# #
# #     gen_images = torch.cat(gen_images)
# #     return gen_images
# #
# #
# # def read_real_images(num_images):
# #     dataloader = InferenceGenerator(create_default_mnist_config(), is_real=True).loader
# #     real_images = []
# #
# #     for (x, y) in tqdm(dataloader, desc="Loading real images"):
# #         real_images.append(x)
# #
# #         if sum([t.shape[0] for t in real_images]) >= num_images:
# #             break
# #
# #     real_images = torch.cat(real_images)[:num_images]
# #     return real_images
#
#
# def compute_fid(real_path: str, gen_path: str, num_images=50000):
#     metric = FrechetInceptionDistance(feature=2048, normalize=True).to('cuda')
#     batch_images_fid = 100
#
#     real_images = read_images(real_path, num_images)
#     gen_images = read_images(gen_path, num_images)
#     print(gen_images.shape, real_images.shape)
#     print(gen_images.dtype, real_images.dtype)
#
#     # assert gen_images.shape == real_images.shape
#
#     for i in tqdm(range(0, gen_images.shape[0], batch_images_fid), desc="FID updating"):
#         metric.update(gen_images[i:i + batch_images_fid].cuda(), real=False)
#         metric.update(real_images[i:i + batch_images_fid].cuda(), real=True)
#
#     result = metric.compute()
#     print(f'FID result: {result}')
#     return result
