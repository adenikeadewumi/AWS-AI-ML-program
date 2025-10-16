

"""First lambda function for serialing image data"""
import json
import boto3
import base64
import traceback

s3 = boto3.client('s3')

def lambda_handler(event, context):
    print(f"Received event from step func: {json.dumps(event)}")
    
    try:
        # Validate input keys
        if "s3_key" not in event or "s3_bucket" not in event:
            raise KeyError("Event must contain 's3_key' and 's3_bucket' fields")
        
        key = event["s3_key"]
        bucket = event["s3_bucket"]

        # Download file from S3
        local_path = "/tmp/image.png"
        try:
            s3.download_file(bucket, key, local_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download file from s3://{bucket}/{key}") from e

        # Read file and encode in base64
        try:
            with open(local_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")  # ensure JSON serializable
        except Exception as e:
            raise RuntimeError("Failed to read or encode image file") from e

        # Successful response
        return {
            "statusCode": 200,
            "body": {
                "image_data": image_data,
                "s3_bucket": bucket,
                "s3_key": key,
                "inferences": []
            }
        }
    
    except Exception as e:
        # Capture stack trace for debugging in CloudWatch
        print("Error:", str(e))
        traceback.print_exc()

        return {
            "statusCode": 500,
            "body": {
                "error": str(e),
                "event": event  # include input for debugging
            }
        }




"""Second  function for  image classifications"""

# Create SageMaker Runtime client
runtime_client = boto3.client('sagemaker-runtime')

# Replace this with your real endpoint name
ENDPOINT = "img-classification-job-2025-10-03-19-15-02-162"

def lambda_handler(event, context):
    try:
        # Decode the image data (coming in base64 format from Step Function or S3 Lambda)
        image = base64.b64decode(event["image_data"])

        # Call SageMaker endpoint using boto3
        response = runtime_client.invoke_endpoint(
            EndpointName=ENDPOINT,
            Body=image,
            ContentType="image/png"
        )

        # Parse response body
        inferences = json.loads(response['Body'].read().decode("utf-8"))

        # Attach inferences to event
        event["inferences"] = inferences

        return {
            "statusCode": 200,
            "body": json.dumps(event)
        }

    except KeyError as e:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": f"Missing key in event: {str(e)}"})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }


"""Third function: to filter off low confidence inferences"""

THRESHOLD = .93
def lambda_handler(event, context):

    # Grab the inferences from the event
    inferences = event["inferences"]

    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = max(list(inferences))>THRESHOLD ## TODO: fill in

    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
