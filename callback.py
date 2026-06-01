import os
import shutil
import glob
from pytorch_lightning.callbacks import Callback

class DriveSyncCallback(Callback):
    def __init__(self, local_dir, drive_dir, sync_every_n_epochs=1):
        """
        Safely copies checkpoints from Colab local disk to Google Drive.
        Resolves local symlinks to prevent the FUSE Errno 95 crash.
        """
        self.local_dir = local_dir
        self.drive_dir = drive_dir
        self.sync_every_n_epochs = sync_every_n_epochs
        
        # Ensure the target drive directory exists
        if not os.path.exists(self.drive_dir):
            os.makedirs(self.drive_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        # Sync only on the specified interval to save network overhead
        if (trainer.current_epoch + 1) % self.sync_every_n_epochs == 0:
            print(f"\n[DriveSyncCallback] Safely syncing checkpoints to Drive at Epoch {trainer.current_epoch}...")
            
            # Grab all checkpoint files from the local directory
            ckpt_files = glob.glob(os.path.join(self.local_dir, "*.ckpt"))
            
            for file_path in ckpt_files:
                filename = os.path.basename(file_path)
                dest_path = os.path.join(self.drive_dir, filename)
                
                try:
                    # THE FIX: If Lightning created a symlink locally (like last.ckpt)
                    # We trace it to the REAL file, and copy the real file to Drive.
                    if os.path.islink(file_path):
                        real_path = os.path.realpath(file_path)
                        shutil.copy2(real_path, dest_path)
                    else:
                        shutil.copy2(file_path, dest_path)
                except Exception as e:
                    print(f"Failed to sync {filename}: {e}")
                    
            print("[DriveSyncCallback] Sync complete!")