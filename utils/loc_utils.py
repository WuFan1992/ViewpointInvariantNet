import os
import numpy as np
import cv2


def calculate_pose_errors(R_gt, t_gt, R_est, t_est):
    # Calculate rotation error
    rotError = np.matmul(R_est.T, R_gt)
    rotError = cv2.Rodrigues(rotError)[0]
    rotError = np.linalg.norm(rotError) * 180 / np.pi

    # Calculate translation error
    transError = np.linalg.norm(t_gt - t_est.squeeze(1)) * 100  # Convert to cm
    #transError = np.linalg.norm(-R_gt.T @ t_gt + R_est.T @ t_est, axis=0)*100
    #transError = np.median(transError)
    
    return rotError, transError

def log_errors(model_path, name, rotation_errors, translation_errors, inplace_text):
    
    total_frames = len(rotation_errors)
    # Remove NaN values from rotation_errors and translation_errors
    rotation_errors = [err for err in rotation_errors if not np.isnan(err)]
    translation_errors = [err for err in translation_errors if not np.isnan(err)]

    # Ensure both lists have the same length after NaN removal
    min_length = min(len(rotation_errors), len(translation_errors))
    rotation_errors = rotation_errors[:min_length]
    translation_errors = translation_errors[:min_length]

    # Update total_frames after NaN removal
    total_frames = len(rotation_errors)
    median_rErr = np.median(rotation_errors)
    median_tErr = np.median(translation_errors)

    # Compute accuracy percentages
    pct10_5 = sum(r <= 5 and t <= 10 for r, t in zip(rotation_errors, translation_errors)) / total_frames * 100
    pct5 = sum(r <= 5 and t <= 5 for r, t in zip(rotation_errors, translation_errors)) / total_frames * 100
    pct2 = sum(r <= 2 and t <= 2 for r, t in zip(rotation_errors, translation_errors)) / total_frames * 100
    pct1 = sum(r <= 1 and t <= 1 for r, t in zip(rotation_errors, translation_errors)) / total_frames * 100

    print('Accuracy:', inplace_text)
    print(f'\t10cm/5deg: {pct10_5:.1f}%')
    print(f'\t5cm/5deg: {pct5:.1f}%')
    print(f'\t2cm/2deg: {pct2:.1f}%')
    print(f'\t1cm/1deg: {pct1:.1f}%')
    print(f'\tmedian_rErr: {median_rErr:.3f} deg')
    print(f'\tmedian_tErr: {median_tErr:.3f} cm')

    # Log median errors to separate files
    log_dir = os.path.join(model_path, 'error_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    with open(os.path.join(log_dir, f'median_error_{name}_{inplace_text}_end.txt'), 'w') as f:
            
        f.write('Accuracy:\n')
        f.write(f'\t10cm/5deg: {pct10_5:.1f}%\n')
        f.write(f'\t5cm/5deg: {pct5:.1f}%\n')
        f.write(f'\t2cm/2deg: {pct2:.1f}%\n')
        f.write(f'\t1cm/1deg: {pct1:.1f}%\n')
        f.write(f'Median translation error: {median_tErr:.6f} cm\n')
        f.write(f'Median rotation error: {median_rErr:.6f} dg\n')