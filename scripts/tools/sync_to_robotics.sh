rsync -avhP \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'logs/' \
  --exclude '.pytest_cache/' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  --exclude '.git/'\
  --exclude '.codegraph/'\
  --exclude 'deep_robotics_model'\
  --exclude 'outputs/'\
  --exclude 'pose_optimization/' \
  --exclude 'third_party/' \
  --exclude '.claude' \
    /home/tuanpm/vqr_stunt_rl/ \
    robotics@100.114.220.18:/home/robotics/tuanpm48/vqr/rl_training/
