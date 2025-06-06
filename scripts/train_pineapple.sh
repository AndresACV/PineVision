#!/bin/bash

# Pineapple Detection Training Launch Script
# Optimized for high annotation density and RTX 3070

set -e  # Exit on any error

# Configuration
PROJECT_ROOT="/mnt/d/repos/mask-r-ccn"
CONFIG_FILE="config/pineapple_training_simple.yaml"
LOG_DIR="outputs/logs"
MODEL_DIR="outputs/models/pineapple_maskrcnn"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Pineapple Detection Training Script${NC}"
echo -e "${BLUE}====================================${NC}"

# Function to print colored messages
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}📋 $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "Not in project root directory. Please run from $PROJECT_ROOT"
    exit 1
fi

print_status "Found project root directory"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    print_error "Virtual environment not found. Please run: uv venv"
    exit 1
fi

print_status "Virtual environment found"

# Activate virtual environment
source .venv/bin/activate
print_status "Virtual environment activated"

# Check if CUDA is available
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())" > /tmp/cuda_check.txt 2>&1
if grep -q "True" /tmp/cuda_check.txt; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown GPU")
    GPU_MEMORY=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')" 2>/dev/null || echo "Unknown")
    print_status "CUDA available - GPU: $GPU_NAME ($GPU_MEMORY)"
else
    print_warning "CUDA not available, will use CPU (training will be very slow)"
fi

# Check if datasets exist
REQUIRED_FILES=(
    "outputs/dataset/annotations_train.json"
    "outputs/dataset/annotations_val.json"
    "outputs/dataset/annotations_test.json"
    "src/data/images"
)

print_info "Checking dataset files..."
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -e "$file" ]; then
        print_error "Required file/directory not found: $file"
        print_error "Please run YOLO to COCO conversion first"
        exit 1
    fi
done
print_status "All dataset files found"

# Check configuration file
if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Configuration file not found: $CONFIG_FILE"
    exit 1
fi
print_status "Configuration file found: $CONFIG_FILE"

# Create output directories
mkdir -p "$LOG_DIR"
mkdir -p "$MODEL_DIR"
print_status "Output directories created"

# Get current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

print_info "Training logs will be saved to: $LOG_FILE"

# Function to run training with proper monitoring
run_training() {
    print_info "Starting training with high annotation density optimization..."
    print_info "Dataset: 176 images, 12,956 annotations (73.61 avg per image)"
    print_info "Hardware: RTX 3070 optimized (batch size 2, mixed precision)"
    print_info "Model: Mask R-CNN with small object detection optimization"
    
    # Display training parameters
    echo ""
    print_info "Training Parameters:"
    echo "   - Config file: $CONFIG_FILE"
    echo "   - Output directory: $MODEL_DIR"
    echo "   - Log file: $LOG_FILE"
    echo "   - Batch size: 2 (optimized for RTX 3070)"
    echo "   - Max iterations: 10,000"
    echo "   - Learning rate: 0.0005"
    echo "   - Mixed precision: Enabled"
    echo "   - Anchor sizes: [16, 32, 64, 128, 256] (small object optimization)"
    echo ""
    
    # Start training
    python src/training/train_pineapple_maskrcnn.py \
        --config-file "$CONFIG_FILE" \
        --num-gpus 1 \
        2>&1 | tee "$LOG_FILE"
    
    return $?
}

# Function to monitor GPU usage during training
monitor_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        print_info "Starting GPU monitoring (will run in background)..."
        
        # Create GPU monitoring log
        GPU_LOG="$LOG_DIR/gpu_monitoring_${TIMESTAMP}.log"
        
        (
            echo "Timestamp,GPU_Util_%,Memory_Used_MB,Memory_Total_MB,Temperature_C" > "$GPU_LOG"
            while true; do
                nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
                    --format=csv,noheader,nounits | \
                    sed "s/^/$(date '+%Y-%m-%d %H:%M:%S'),/" >> "$GPU_LOG"
                sleep 30
            done
        ) &
        
        GPU_MONITOR_PID=$!
        print_status "GPU monitoring started (PID: $GPU_MONITOR_PID)"
        print_info "GPU logs: $GPU_LOG"
    else
        print_warning "nvidia-smi not found, skipping GPU monitoring"
    fi
}

# Function to cleanup background processes
cleanup() {
    if [ ! -z "$GPU_MONITOR_PID" ]; then
        print_info "Stopping GPU monitoring..."
        kill $GPU_MONITOR_PID 2>/dev/null || true
    fi
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Ask user if they want to run validation first
read -p "Run training setup validation first? (recommended) [Y/n]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    print_info "Running training setup validation..."
    python scripts/test_training_setup.py
    
    if [ $? -ne 0 ]; then
        print_error "Training setup validation failed!"
        read -p "Continue anyway? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_error "Training aborted"
            exit 1
        fi
    else
        print_status "Training setup validation passed!"
    fi
fi

# Start GPU monitoring
monitor_gpu

# Run training
print_info "Starting training process..."
echo "=============================="

# Check if user wants to resume from checkpoint
if [ -f "$MODEL_DIR/last_checkpoint" ]; then
    print_warning "Found existing checkpoint"
    read -p "Resume from checkpoint? [Y/n]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        print_info "Resuming from checkpoint..."
        python src/training/train_pineapple_maskrcnn.py \
            --config-file "$CONFIG_FILE" \
            --resume \
            --num-gpus 1 \
            2>&1 | tee -a "$LOG_FILE"
        TRAINING_RESULT=$?
    else
        print_info "Starting fresh training..."
        run_training
        TRAINING_RESULT=$?
    fi
else
    run_training
    TRAINING_RESULT=$?
fi

# Check training result
if [ $TRAINING_RESULT -eq 0 ]; then
    print_status "Training completed successfully!"
    
    # Show training summary
    echo ""
    print_info "Training Summary:"
    echo "   - Model saved to: $MODEL_DIR"
    echo "   - Training logs: $LOG_FILE"
    
    if [ ! -z "$GPU_MONITOR_PID" ]; then
        echo "   - GPU monitoring: $GPU_LOG"
    fi
    
    # Check if final model exists
    if [ -f "$MODEL_DIR/model_final.pth" ]; then
        MODEL_SIZE=$(du -h "$MODEL_DIR/model_final.pth" | cut -f1)
        print_status "Final model: $MODEL_DIR/model_final.pth ($MODEL_SIZE)"
    fi
    
    # Show next steps
    echo ""
    print_info "Next Steps:"
    echo "1. Check training metrics in TensorBoard:"
    echo "   tensorboard --logdir $MODEL_DIR"
    echo ""
    echo "2. Run inference on test set:"
    echo "   python src/inference/test_model.py --model $MODEL_DIR/model_final.pth"
    echo ""
    echo "3. View training logs:"
    echo "   less $LOG_FILE"
    
else
    print_error "Training failed with exit code $TRAINING_RESULT"
    print_info "Check logs for details: $LOG_FILE"
    exit $TRAINING_RESULT
fi

print_status "Training script completed!" 