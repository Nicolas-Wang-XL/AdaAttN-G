python test.py ^
--content_path datasets/contents ^
--style_path datasets/styles ^
--name artvggGanSelfAdaAttN_without_artvgg_all_adain_with_gan_tt3_512_3 --model artvggganselfadaattn ^
--dataset_mode unaligned ^
--load_size 512 ^
--crop_size 512 ^
--style_encoder_path ./checkpoints\artvgg19_2200_24class_with_norm/latest_net_artvgg.pth ^
--gpu_ids 0 ^
--skip_connection_3 ^
--shallow_layer ^
--data_norm
