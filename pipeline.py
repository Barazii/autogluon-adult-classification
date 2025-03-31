from dotenv import load_dotenv
from sagemaker.workflow.steps import CacheConfig
from sagemaker.workflow.pipeline_context import PipelineSession
import os
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor


def pipeline():
    load_dotenv()

    cache_config = CacheConfig(enable_caching=False, expire_after="10d")

    sagemaker_session = PipelineSession(
        default_bucket=os.environ["S3_BUCKET_NAME"],
    )

    sklprocessor = SKLearnProcessor(
        framework_version=os.environ["SKL_VERSION"],
        instance_type=os.environ["PROCESSING_INSTANCE_TYPE"],
        instance_count=1,
        role=os.environ["SM_EXEC_ROLE"],
        sagemaker_session=sagemaker_session,
        env={
            "PC_BASE_DIR": os.environ["PC_BASE_DIR"],
        }
    )

    # defining the processing step
    processing_step = ProcessingStep(
        name="process-data",
        processor=sklprocessor,
        display_name="process data",
        description="This step is to make the dataset.",
        inputs=[
            ProcessingInput(
                source=os.path.join(os.environ["S3_PROJECT_URI"], "rawdata"),
                destination=os.path.join(os.environ["PC_BASE_DIR"], "rawdata"),
                input_name="rawdata",
                s3_data_type="S3Prefix",
                s3_input_mode="File",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source=os.path.join(os.environ["PC_BASE_DIR"], "train"),
                destination=os.path.join(
                    os.environ["S3_PROJECT_URI"], "processing-step/train"
                ),
                output_name="train",
            ),
            ProcessingOutput(
                source=os.path.join(os.environ["PC_BASE_DIR"], "test"),
                destination=os.path.join(
                    os.environ["S3_PROJECT_URI"], "processing-step/test"
                ),
                output_name="test",
            ),
        ],
        code="src/processing.py",
        cache_config=cache_config,
    )


if __name__ == "__main__":
    pipeline()
