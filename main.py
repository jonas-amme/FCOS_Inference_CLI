import typer
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.panel import Panel
from rich import box
import torch
from utils.inference import (
    Torchvision_Inference,
    ImageProcessor,
    InferenceConfig,
    PatchConfig,
    ProcessorConfig
)
from utils.factory import ModelFactory, ConfigCreator, ModelConfig

# Initialize Typer app
app = typer.Typer(help="CLI for running object detection inference on medical images using trained models.")
console = Console()

def load_model_from_config(config_path: Path):
    """Load model from configuration file."""
    try:
        config = ConfigCreator.load(str(config_path))
        model = ModelFactory.load(config)
        return model, config
    except Exception as e:
        console.print(f"[red]Error loading model: {str(e)}[/red]")
        raise typer.Exit(1)

def setup_inference(
    model: torch.nn.Module,
    is_wsi: bool,
    batch_size: int,
    num_workers: int,
    device: str,
    patch_size: int,
    overlap: float
) -> tuple:
    """Setup inference components."""
    inference_config = InferenceConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        is_wsi=is_wsi
    )

    patch_config = PatchConfig(
        size=patch_size,
        overlap=overlap
    )

    strategy = Torchvision_Inference(model, inference_config)
    processor = ImageProcessor(
        strategy,
        ProcessorConfig()
    )

    return processor, patch_config


def print_model_config(config: ModelConfig) -> None:
    """Print model configuration in a nicely formatted table."""
    # Create main configuration table
    main_table = Table(
        title="[bold]Model Configuration",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )

    # Add columns
    main_table.add_column("Parameter", style="bold green")
    main_table.add_column("Value", style="yellow")

    # Add basic configuration rows
    basic_params = [
        ("Model Name", config.model_name),
        ("Detector Type", config.detector),
        ("Backbone", config.backbone),
        ("Number of Classes", config.num_classes),
        ("Detection Threshold", f"{config.det_thresh:.3f}"),
        ("Extra Blocks", str(config.extra_blocks)),
        ("Patch Size", str(config.patch_size)),
    ]

    for param, value in basic_params:
        main_table.add_row(param, str(value))

    # Add optional parameters if they exist
    if hasattr(config, 'weights') and config.weights:
        main_table.add_row("Initial Weights", str(config.weights))

    if hasattr(config, 'returned_layers') and config.returned_layers:
        main_table.add_row("Returned Layers", str(config.returned_layers))

    # Special handling for detector-specific parameters
    if hasattr(config, 'anchor_sizes') and config.anchor_sizes:
        main_table.add_row("Anchor Sizes", str(config.anchor_sizes))

    if hasattr(config, 'anchor_ratios') and config.anchor_ratios:
        main_table.add_row("Anchor Ratios", str(config.anchor_ratios))

    # Print the configuration
    console.print("\n")
    console.print(Panel(main_table, title="[bold blue]Model Configuration",
                       subtitle="[bold blue]Detection Pipeline"))
    console.print("\n")



@app.command()
def detect(
    config_path: Path = typer.Argument(
        ...,
        help="Path to model configuration YAML file",
        exists=True
    ),
    input_path: Path = typer.Argument(
        ...,
        help="Path to input image or directory"
    ),
    output_dir: Path = typer.Argument(
        ...,
        help="Directory to save detection results"
    ),
    is_wsi: bool = typer.Option(
        False,
        "--wsi",
        help="Process whole slide images"
    ),
    batch_size: int = typer.Option(
        8,
        "--batch-size",
        "-b",
        help="Batch size for inference"
    ),
    num_workers: int = typer.Option(
        4,
        "--workers",
        "-w",
        help="Number of worker processes"
    ),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        "--device",
        "-d",
        help="Device for inference"
    ),
    patch_size: int = typer.Option(
        1024,
        "--patch-size",
        "-p",
        help="Size of image patches"
    ),
    overlap: float = typer.Option(
        0.3,
        "--overlap",
        "-o",
        help="Overlap between patches"
    )
) -> None:
    """
    Perform object detection on images using a trained model.
    """
    try:
        # Load model
        with console.status("[bold green]Loading model...") as status:
            model, config = load_model_from_config(config_path)
            status.update("[bold green]Model loaded successfully!")

        # Print configuration
        print_model_config(config)

        # Print runtime settings
        runtime_table = Table(
            title="[bold]Runtime Settings",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )
        runtime_table.add_column("Parameter", style="bold green")
        runtime_table.add_column("Value", style="yellow")

        runtime_params = [
            ("Processing Mode", "Whole Slide Image" if is_wsi else "Regular Image"),
            ("Batch Size", str(batch_size)),
            ("Number of Workers", str(num_workers)),
            ("Device", device),
            ("Patch Size", str(patch_size)),
            ("Patch Overlap", f"{overlap:.2f}"),
            ("Input Path", str(input_path)),
            ("Output Directory", str(output_dir))
        ]

        for param, value in runtime_params:
            runtime_table.add_row(param, value)

        console.print(Panel(runtime_table, title="[bold blue]Runtime Configuration"))
        console.print("\n")

        # Setup inference components
        processor, patch_config = setup_inference(
            model, is_wsi, batch_size, num_workers,
            device, patch_size, overlap
        )

        # Define valid extensions based on image type
        wsi_extensions = ('.svs', '.tif', '.tiff', '.dcm', '.vms', '.ndpi', '.vmu', '.mrxs', '.czi')
        regular_extensions = ('.tif', '.tiff', '.jpg', '.jpeg', '.png')
        valid_extensions = wsi_extensions if is_wsi else regular_extensions

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process images
        if input_path.is_file():
            # Check if file extension is valid
            if input_path.suffix.lower() not in valid_extensions:
                console.print(f"[red]Invalid file type! Expected one of: {', '.join(valid_extensions)}[/red]")
                raise typer.Exit(1)

            # Single image processing
            console.print(f"\n[bold]Processing single image: {input_path.name}[/bold]")
            processor.process_single(
                input_path,
                patch_config=patch_config,
                output_dir=output_dir
            )
            console.print(f"[green]Results saved to: {output_dir / f'{input_path.stem}_detections.json'}[/green]")

        elif input_path.is_dir():
            # Batch processing
            image_paths = []
            for ext in valid_extensions:
                image_paths.extend(input_path.glob(f"*{ext}"))
                image_paths.extend(input_path.glob(f"*{ext.upper()}"))  # Handle uppercase extensions

            if not image_paths:
                console.print(f"[red]No {'WSI' if is_wsi else 'regular'} images found in directory![/red]")
                console.print(f"[red]Supported extensions: {', '.join(valid_extensions)}[/red]")
                raise typer.Exit(1)

            console.print(f"\n[bold]Found {len(image_paths)} valid images[/bold]")
            console.print(f"[bold]Processing {len(image_paths)} images...[/bold]")
            processor.process_multi(
                image_paths,
                patch_config=patch_config,
                output_dir=output_dir
            )
            console.print(f"[green]Results saved to: {output_dir}[/green]")

        else:
            console.print("[red]Invalid input path![/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error during processing: {str(e)}[/red]")
        raise typer.Exit(1)
    
@app.command()
def validate_config(
    config_path: Path = typer.Argument(
        ...,
        help="Path to model configuration YAML file",
        exists=True
    )
) -> None:
    """
    Validate a model configuration file.
    """
    try:
        with console.status("[bold green]Validating configuration...") as status:
            config = ConfigCreator.load(str(config_path))
            status.update("[bold green]Configuration loaded successfully!")

        print_model_config(config)
        console.print("[green]✓ Configuration is valid![/green]")

    except Exception as e:
        console.print(f"[red]Invalid configuration: {str(e)}[/red]")
        raise typer.Exit(1)
if __name__ == "__main__":
    app()