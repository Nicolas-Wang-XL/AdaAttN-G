python test.py ^
--content_path datasets/contents ^
--style_path datasets/styles ^
--name AdaAttN_without_edge --model adaattn ^
--dataset_mode unaligned ^
--load_size 512 ^
--crop_size 512 ^
--image_encoder_path checkpoints/vgg_normalised.pth ^
--gpu_ids 0 ^
--skip_connection_3 ^
--shallow_layer
