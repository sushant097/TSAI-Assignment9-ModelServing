
train: ## Train the model
	python ../src/train_script.py experiment=example_timm trainer=gpu

debug: ## Enter debugging mode with pdb
	#
	# tips:
	# - use "import pdb; pdb.set_trace()" to set breakpoint
	# - use "h" to print all commands
	# - use "n" to execute the next line
	# - use "c" to run until the breakpoint is hit
	# - use "l" to print src code around current line, "ll" for full function code
	# - docs: https://docs.python.org/3/library/pdb.html
	#
	python -m pdb src/train.py debug=default

# Makefile for docker build
build:
	docker build -t deploy dockerize/

run:
    docker run -t -p 8080:8080 deploy:latest
