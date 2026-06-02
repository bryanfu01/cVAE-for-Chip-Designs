import os
import shutil
import glob
from pytorch_lightning.callbacks import Callback

class DriveSyncCallback(Callback):
    def __init__(self, drive_dir, sync_every_n_epochs=10):
        """
        Safely copies checkpoints from Colab local disk to Google Drive.
        Resolves local symlinks to prevent the FUSE Errno 95 crash.
        """
        self.drive_dir = drive_dir
        self.sync_every_n_epochs = sync_every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        # Sync only on the specified interval to save network overhead
        if (trainer.current_epoch + 1) % self.sync_every_n_epochs == 0:
            print(f"\n[DriveSyncCallback] Safely syncing checkpoints to Drive at Epoch {trainer.current_epoch}...")
            
            # DYNAMIC FIX: Ask Lightning exactly where it is saving the checkpoints!
            local_ckpt_dir = trainer.checkpoint_callback.dirpath
            
            # Create a matching version folder in Drive so runs stay organized
            version_folder = f"version_{trainer.logger.version}"
            target_drive_dir = os.path.join(self.drive_dir, version_folder, "checkpoints")
            os.makedirs(target_drive_dir, exist_ok=True)
            
            # Prevent Drive Bloat: Delete old checkpoints for this specific run before copying new ones
            existing_drive_ckpts = glob.glob(os.path.join(target_drive_dir, "*.ckpt"))
            for old_ckpt in existing_drive_ckpts:
                try:
                    os.remove(old_ckpt)
                except Exception:
                    pass
            
            # Grab all current checkpoint files from the correct local directory
            ckpt_files = glob.glob(os.path.join(local_ckpt_dir, "*.ckpt"))
            
            for file_path in ckpt_files:
                filename = os.path.basename(file_path)
                dest_path = os.path.join(target_drive_dir, filename)
                
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
                    
            print(f"[DriveSyncCallback] Sync complete to {target_drive_dir}!")