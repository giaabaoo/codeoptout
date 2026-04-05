from code_takedown.pipelines import get_pipeline
from code_takedown.utils import launch, setup_config


def main(config):
    pipeline = get_pipeline(config)
    pipeline.run()

if __name__ == '__main__':
    config = setup_config()
    # # setup_logger(config)
    
    # launch(
    #     main,
    #     config.num_gpus,
    #     num_machines=config.num_machines,
    #     machine_rank=config.machine_rank,
    #     dist_url='auto',
    #     args=(config, ),
    # )

    main(config)

