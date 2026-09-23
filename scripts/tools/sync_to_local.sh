rsync -avhP \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'logs/' \
  --exclude '.pytest_cache/' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  --exclude '.git/'\
  /home/robotics/tuanpm48/vqr/rl_training/ \
  tuanpm@<SERVER_IP>:~/tuanpm48/vqr/rl_training/