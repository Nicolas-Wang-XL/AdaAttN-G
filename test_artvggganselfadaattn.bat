python test.py ^
--content_path datasets/contents_all ^
--style_path datasets/stylest ^
--name artvggGanSelfAdaAttN_without_artvgg_with_artmap_all_adain_with_gan_monet_256 --model artvggganselfadaattn ^
--dataset_mode unaligned ^
--load_size 256 ^
--crop_size 256 ^
--style_encoder_path ./checkpoints\artvgg19_2200_24class_with_norm/latest_net_artvgg.pth ^
--gpu_ids 0 ^
--skip_connection_3 ^
--shallow_layer ^
--data_norm ^
--art_map
