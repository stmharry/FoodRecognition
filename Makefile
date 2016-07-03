ROOT_DIR=/mnt/data/Foods
TRAIN_DIR=$(ROOT_DIR)/dish-clean
TEST_DIR=$(ROOT_DIR)/dish-test
WORKING_DIR=$(ROOT_DIR)/dish-clean-save1
NUM_TEST_CROPS=4

train:
	./resnet_main.py \
		--command train \
		--train_dir $(TRAIN_DIR) \
		--working_dir $(WORKING_DIR) \
		--train_iteration 10000 \
		--lr_half_per 1500

test:
	./resnet_main.py \
		--command test \
		--train_dir $(TRAIN_DIR) \
		--test_dir $(TEST_DIR) \
		--working_dir $(WORKING_DIR) \
		--log_file $(ROOT_DIR)/test_log.csv \
		--num_test_crops 4

feat-train:
	./resnet_main.py \
		--command test \
		--test_attrs key,label,feat \
		--train_dir $(TRAIN_DIR) \
		--test_dir $(TRAIN_DIR) \
		--working_dir $(WORKING_DIR) \
		--log_file $(ROOT_DIR)/train_feat_log.csv \
		--num_test_crops 16

feat-test:
	./resnet_main.py \
		--command test \
		--test_attrs key,label,feat \
		--train_dir $(TRAIN_DIR) \
		--test_dir $(TEST_DIR) \
		--working_dir $(WORKING_DIR) \
		--log_file $(ROOT_DIR)/test_feat_log.csv \
		--num_test_crops 16

check:
	./check_data.py \
		--train_dir $(TRAIN_DIR)

tree:
	./make_feat.py \
		--make_tree \
		--log_file $(LOG_FILE) \
		--tree_file $(ROOT_DIR)/tree
