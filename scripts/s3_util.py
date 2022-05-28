import boto3
import os

def download_npz_dir(bucket, poi_list, out_path):
    s3 = boto3.client('s3')
    for poi in poi_list:
        lab_path = 'npz/labels/' + poi
        planet_path = 'npz/planet/' + poi
        for path in [lab_path, planet_path]:
            obj_list = s3.list_objects_v2(
                Bucket = bucket,
                Prefix = path
            )
            keys = [item['Key'] for item in obj_list['Contents']]
            dir_ = keys[0]
            files = keys[1:] # drop the first dir key
            for file in files:
                dir_path = os.path.join(out_path, dir_)
                os.makedirs(dir_path, exist_ok=True)
                file_path = os.path.join(out_path, file)
                with open(file_path, 'wb') as data:
                    s3.download_fileobj(bucket, file, data)

if __name__ == '__main__':
    download_npz_dir('capp-advml-land-use-prediction-small', ['2697_3715_13_20N', '5989_3554_13_44N'], 'tst')
