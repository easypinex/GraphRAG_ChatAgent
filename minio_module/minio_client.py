from minio import Minio
from minio.error import S3Error
from io import BytesIO
import json
from mimetypes import MimeTypes

class MinioClient:
    def __init__(self, **clinet_kargs):
        self.client = Minio(**clinet_kargs)

    def upload_file_to_minio(self, bucket: str, source_file_path: str, destination_file_path: str) -> bool:
        """
        Upload a file to MinIO.

        Parameters:
            bucket (str): Name of the destination bucket.
            source_file_path (str): Path of the source file to upload.
            destination_file_path (str): Path where the file will be saved in MinIO.

        Returns:
            bool: True if upload is successful, False otherwise.
        """

        try:
            # Check if the bucket exists, create it if it doesn't
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)
                print(f"Bucket '{bucket}' created successfully.")
            else:
                print(f"Bucket '{bucket}' already exists.")

            # Upload the file
            content_type = MimeTypes().guess_type(source_file_path)[0]
            if content_type == 'application/json':
                content_type = "text/plain" # for minio console preview
            self.client.fput_object(bucket, destination_file_path, source_file_path, content_type=content_type)
            print(f"File '{source_file_path}' uploaded to '{bucket}/{destination_file_path}'.")
            return True

        except S3Error as exc:
            print("An error occurred while uploading to MinIO:", exc)
            return False

    def download_file_from_minio(self, bucket: str, source_file_path: str, destination_file_path: str) -> bool:
        """
        Download a file from MinIO.

        Parameters:
            bucket (str): Name of the source bucket.
            source_file_path (str): Path of the file in MinIO to download.
            destination_file_path (str): Local path to save the downloaded file.

        Returns:
            bool: True if download is successful, False otherwise.
        """
        try:
            # Download the file
            self.client.fget_object(bucket, source_file_path, destination_file_path)
            print(f"File '{source_file_path}' downloaded from '{bucket}' to '{destination_file_path}'.")
            return True

        except S3Error as exc:
            print("An error occurred while downloading from MinIO:", exc)
            return False
        
    def upload_dict_as_json(self, bucket: str, json_data: dict|list, destination_file_path: str) -> bool:
        """
        Upload a dictionary as a JSON file directly to MinIO.

        Parameters:
            bucket (str): Name of the destination bucket.
            json_data (dict): Dictionary data to upload.
            destination_file_path (str): Path where the JSON will be saved in MinIO.

        Returns:
            bool: True if upload is successful, False otherwise.
        """
        try:
            # Convert dictionary to JSON string and encode to bytes
            json_str = json.dumps(json_data, ensure_ascii=False, indent=2, sort_keys=True)
            json_bytes = BytesIO(json_str.encode('utf-8'))
            size = json_bytes.getbuffer().nbytes

            # Check if the bucket exists, create it if it doesn't
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)
                print(f"Bucket '{bucket}' created successfully.")
            else:
                print(f"Bucket '{bucket}' already exists.")

            # Upload the JSON file
            self.client.put_object(bucket, destination_file_path, json_bytes, size, content_type="text/plain") # "text/plain" is for minio console preview
            print(f"Dictionary uploaded as JSON to '{bucket}/{destination_file_path}'.")
            return True

        except Exception as exc:
            print("An error occurred while uploading JSON to MinIO:", exc)
            return False
        
    def download_json_as_dict(self, bucket: str, source_file_path: str) -> dict | list | None:
        """
        Download a JSON file from MinIO and return it as a dictionary or list.

        Parameters:
            bucket (str): Name of the bucket containing the file.
            source_file_path (str): Path of the JSON file in the bucket.

        Returns:
            dict | list | None: The content of the JSON file as a dictionary or list, or None if an error occurs.
        """
        try:
            # Get the object from MinIO
            response = self.client.get_object(bucket, source_file_path)

            # Read and decode the content of the response
            json_bytes = BytesIO(response.read())
            json_data = json.loads(json_bytes.getvalue().decode('utf-8'))

            print(f"JSON file '{source_file_path}' downloaded and parsed successfully.")
            return json_data

        except Exception as exc:
            print("An error occurred while downloading JSON from MinIO:", exc)
            return None

    def check_file_exists(self, bucket: str, file_path: str) -> bool:
        """
        Check if a file exists in the specified bucket.

        Parameters:
            bucket (str): Name of the bucket.
            file_path (str): Path of the file to check.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        try:
            # Stat object to check if the file exists and get its information
            file_stat = self.client.stat_object(bucket, file_path)
            # print("File exists. Details:")
            # print(f"Name: {file_stat.object_name}")
            # print(f"Size: {file_stat.size} bytes")
            # print(f"Last Modified: {file_stat.last_modified}")
            # print(f"ETag: {file_stat.etag}")
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                print(f"File '{file_path}' does not exist in bucket '{bucket}'.")
            else:
                print(f"An error occurred while checking the file: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False
        
    def list_files(self, bucket: str, prefix: str = "") -> list[str]:
        return [obj.object_name for obj in self.client.list_objects(bucket, prefix=prefix, recursive=False)]
        
# Example usage
if __name__ == "__main__":
    minio_client = MinioClient(endpoint="localhost:9000", 
                               access_key="XYiGeY45XzQYcEkOLItK", 
                               secret_key="33ycWL0FfBrqEE4tUjtmX7uJBW4vlfSidnftmeMU", 
                               secure=False)
    bucket_name = "llm-graph-bucket"
    source_file = "test/test_data/serialization/duplicate_nodes.json"  # File to upload
    destination_file = "test/duplicate_nodes.json"  # Destination in MinIO
    download_path = "test/test_data/serialization/duplicate_nodes_from_download.json"  # Local path to download file

    # Upload file
    if minio_client.upload_file_to_minio(bucket_name, source_file, destination_file):
        print("File uploaded successfully.")
    else:
        print("File upload failed.")

    # Download file
    if minio_client.download_file_from_minio(bucket_name, destination_file, download_path):
        print("File downloaded successfully.")
    else:
        print("File download failed.")