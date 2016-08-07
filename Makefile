ROOT_DIR=/mnt/data/Foods
TRAIN_DIR=$(ROOT_DIR)/dish-clean
TEST_DIR=$(ROOT_DIR)/dish-test
WORKING_DIR=$(ROOT_DIR)/dish-clean-save-1.0

LOG=$(ROOT_DIR)/test_log.csv
TRAIN_FEAT_LOG=$(ROOT_DIR)/train_feat_log.csv
TEST_FEAT_LOG=$(ROOT_DIR)/test_feat_log.csv
NUM_TEST_CROPS=4

protoc:
	protoc --python_out=. --grpc_out=. --plugin=protoc-gen-grpc=`which grpc_python_plugin` example.proto

debug:
	python -i resnet_main.py \
		--train_dir $(TRAIN_DIR) \
		--working_dir $(WORKING_DIR) \

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
		--log_file $(LOG) \
		--num_test_crops 4

feat-train:
	./resnet_main.py \
		--command test \
		--test_attrs key,label,feat \
		--train_dir $(TRAIN_DIR) \
		--test_dir $(TRAIN_DIR) \
		--working_dir $(WORKING_DIR) \
		--log_file $(TRAIN_FEAT_LOG) \
		--num_test_crops 16

feat-test:
	./resnet_main.py \
		--command test \
		--test_attrs key,label,feat \
		--train_dir $(TRAIN_DIR) \
		--test_dir $(TEST_DIR) \
		--working_dir $(WORKING_DIR) \
		--log_file $(TEST_FEAT_LOG) \
		--num_test_crops 16

check:
	./check_data.py \
		--train_dir $(TRAIN_DIR)

tree:
	./make_tree.py \
		--train_log_file $(TRAIN_FEAT_LOG) \
		--test_log_file $(TEST_FEAT_LOG) \
		--image_dir $(ROOT_DIR)/tree
