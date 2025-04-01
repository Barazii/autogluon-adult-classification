from dotenv import load_dotenv
from sagemaker.workflow.steps import CacheConfig
from sagemaker.workflow.pipeline_context import PipelineSession
import os
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.estimator import Estimator
from sagemaker import image_uris, model_uris, script_uris, hyperparameters
from sagemaker.workflow.steps import TrainingStep
from sagemaker.model import Model
from sagemaker.workflow.model_step import ModelStep
from sagemaker.inputs import TrainingInput
from sagemaker.transformer import Transformer
from sagemaker.workflow.steps import TransformStep
from sagemaker.workflow.functions import Join
from pathlib import Path
import tempfile
import subprocess
import boto3


def pipeline():
    load_dotenv(override=True)

    cache_config = CacheConfig(enable_caching=True, expire_after="10d")

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
        },
    )

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
                source=os.path.join(os.environ["PC_BASE_DIR"], "training"),
                destination=os.path.join(
                    os.environ["S3_PROJECT_URI"], "processing-step/training"
                ),
                output_name="training",
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

    model_id = "autogluon-classification-ensemble"
    model_version = "*"
    train_scope = "training"

    train_image_uri = image_uris.retrieve(
        region="eu-north-1",
        framework=None,
        model_id=model_id,
        model_version="*",
        image_scope=train_scope,
        instance_type=os.environ["TRAINING_INSTANCE_TYPE"],
    )
    train_source_uri = script_uris.retrieve(
        model_id=model_id,
        model_version=model_version,
        script_scope=train_scope,
    )
    train_model_uri = model_uris.retrieve(
        model_id=model_id,
        model_version=model_version,
        model_scope=train_scope,
    )
    hp = hyperparameters.retrieve_default(
        model_id=model_id, model_version=model_version
    )
    hp["auto_stack"] = "True"

    estimator = Estimator(
        image_uri=train_image_uri,
        source_dir=train_source_uri,
        model_uri=train_model_uri,
        entry_point="transfer_learning.py",
        role=os.environ["SM_EXEC_ROLE"],
        max_run=360000,
        instance_count=int(os.environ["TRAINING_INSTANCE_COUNT"]),
        instance_type=os.environ["TRAINING_INSTANCE_TYPE"],
        input_mode="File",
        base_job_name="training_job",
        sagemaker_session=sagemaker_session,
        hyperparameters=hp,
    )

    training_step = TrainingStep(
        name="transfer-learning",
        step_args=estimator.fit(
            inputs={
                "training": TrainingInput(
                    s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
                        "training"
                    ].S3Output.S3Uri,
                    content_type="text/csv",
                    s3_data_type="S3Prefix",
                ),
            },
        ),
        cache_config=cache_config,
    )

    inference_scope = "inference"
    inference_image_uri = image_uris.retrieve(
        region=os.environ["AWS_REGION"],
        framework=None,
        image_scope=inference_scope,
        model_id=model_id,
        model_version=model_version,
        instance_type=os.environ["INFERENCE_INSTANCE_TYPE"],
    )
    inference_sourcedir_uri = script_uris.retrieve(
        model_id=model_id,
        model_version=model_version,
        script_scope=inference_scope,
    )
    # download sourcedir, upalod to s3 (otherwise error for some reason)
    tmp_dir = Path(tempfile.mkdtemp())
    sourcedir = tmp_dir / "sourcedir"
    sourcedir.mkdir(exist_ok=True)
    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            inference_sourcedir_uri,
            f"{sourcedir}/",
        ],
        check=True,
    )
    s3 = boto3.client("s3")
    s3.upload_file(
        f"{sourcedir}/sourcedir.tar.gz",
        os.environ["S3_BUCKET_NAME"],
        "sourcedir.tar.gz",
    )

    prediction_model = Model(
        image_uri=inference_image_uri,
        source_dir=os.path.join(os.environ["S3_PROJECT_URI"], "sourcedir.tar.gz"),
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        entry_point="inference.py",
        role=os.environ["SM_EXEC_ROLE"],
        name="autogluon-adult-classification-ensemble",
        sagemaker_session=sagemaker_session,
    )

    from sagemaker.sklearn.model import SKLearnModel
    from sagemaker.pipeline import PipelineModel

    inference_parsing_model = SKLearnModel(
        name="inference-parsing-model",
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        entry_point=f"src/inference_parsing.py",
        framework_version=os.environ["SKL_VERSION"],
        sagemaker_session=sagemaker_session,
        role=os.environ["SM_EXEC_ROLE"],
    )

    model = PipelineModel(
        name="model",
        models=[prediction_model, inference_parsing_model],
        sagemaker_session=sagemaker_session,
        role=os.environ["SM_EXEC_ROLE"],
    )

    model_step = ModelStep(
        name="create-model",
        step_args=model.create(instance_type=os.environ["PROCESSING_INSTANCE_TYPE"]),
    )

    # define the transformer for evaluation
    transformer = Transformer(
        model_name=model_step.properties.ModelName,
        instance_type=os.environ["TRANSFORM_INSTANCE_TYPE"],
        instance_count=int(os.environ["TRANSFORM_INSTANCE_COUNT"]),
        strategy="MultiRecord",
        accept="text/csv",
        assemble_with="Line",
        output_path=os.environ["S3_TRANSFORM_OUTPUT_URI"],
        sagemaker_session=sagemaker_session,
    )
    transform_step = TransformStep(
        name="make-test-predictions",
        description="This step generates predictions on the test data using the trained model for evaluation.",
        step_args=transformer.transform(
            data=Join(
                on="/",
                values=[
                    processing_step.properties.ProcessingOutputConfig.Outputs[
                        "test"
                    ].S3Output.S3Uri,
                    "test_without_labels.csv",
                ],
            ),
            # data="s3://autogluon-adult-classification/processing-step/test/test_without_labels.csv",
            split_type="Line",
            join_source="None",
            content_type="text/csv",
        ),
        cache_config=cache_config,
    )

    # build the pipeline
    pipeline = Pipeline(
        name="autogluon-adult-classification-pipeline",
        steps=[processing_step, training_step, model_step],
        sagemaker_session=sagemaker_session,
    )

    pipeline.upsert(
        role_arn=os.environ["SM_EXEC_ROLE"],
        description="A pipeline to train the AWS autogluon model on the adult dataset.",
    )

    pipeline.start()


if __name__ == "__main__":
    pipeline()
