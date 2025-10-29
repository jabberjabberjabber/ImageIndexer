$host.UI.RawUI.WindowTitle = "Indexer Launcher"

$resourcesDir = Join-Path $PSScriptRoot "../resources"
$koboldArgsPath = Join-Path $resourcesDir "kobold_args.json"

function Show-Menu {
    param (
        [string]$Title = 'Indexer Launcher'
    )
    Clear-Host
    Write-Host "================ $Title ================" -ForegroundColor Cyan
    Write-Host ""
	Write-Host "1: " -ForegroundColor Yellow -NoNewline; Write-Host "Install Requirements" -ForegroundColor Green
    Write-Host "2: " -ForegroundColor Yellow -NoNewline; Write-Host "Run Indexer with Model" -ForegroundColor Green
    Write-Host "3: " -ForegroundColor Yellow -NoNewline; Write-Host "Run Indexer Alone" -ForegroundColor Green
	Write-Host "4: " -ForegroundColor Yellow -NoNewline; Write-Host "Select Model" -ForegroundColor Green

    Write-Host "Q: " -ForegroundColor Yellow -NoNewline; Write-Host "Quit" -ForegroundColor Red
    Write-Host ""
}

function Run-WithAI {
    try {
        if (Test-Path $koboldArgsPath) {
            Write-Host "Loading configuration..." -ForegroundColor Cyan
            $koboldArgs = Get-Content $koboldArgsPath -Raw | ConvertFrom-Json
            
            $executablePath = Join-Path $resourcesDir $koboldArgs.executable
            $commandArgs = @(
                $koboldArgs.model_param,
                "--mmproj", $koboldArgs.mmproj,
                "--contextsize", $koboldArgs.contextsize,
                "--visionmaxres", $koboldArgs.visionmaxres,
				"--chatcompletionsadapter", $koboldArgs.chatcompletionsadapter
            )

            # Add flashattention flag if enabled
            if ($koboldArgs.flashattention -eq $true) {
                $commandArgs += "--flashattention"
            }
            
            Write-Host "Starting Indexer with AI support..." -ForegroundColor Green
            Write-Host "Executable: " -NoNewline -ForegroundColor Gray
            Write-Host "$executablePath" -ForegroundColor White
            
			# Set the working directory to the koboldcpp exec directory so that it will save model files there
			$workingDir = Split-Path -Path $executablePath -Parent
			# Call koboldcpp
            Start-Process -FilePath $executablePath -ArgumentList $commandArgs -NoNewWindow -workingDir $workingDir
            
            Run-GUI
        } else {
            Write-Host "Error: Kobold arguments file not found at $koboldArgsPath" -ForegroundColor Red
            Read-Host "Press Enter to continue..." | Out-Null
        }
    } catch {
        Write-Host "Error running indexer with AI: $_" -ForegroundColor Red
        Read-Host "Press Enter to continue..." | Out-Null
    }
}

function Run-Alone {
    try {
        Write-Host "Running Indexer alone..." -ForegroundColor Green
        Run-GUI
    } catch {
        Write-Host "Error running indexer alone: $_" -ForegroundColor Red
        Read-Host "Press Enter to continue..." | Out-Null
    }
}

function Run-GUI {
	try {
        Write-Host "Launching Python GUI component..." -ForegroundColor Cyan
		start-process "src\gui.bat"
        } catch {
            Write-Host "GUI component not found. Please check installation." -ForegroundColor Red
            Write-Host "Expected path: " -NoNewline -ForegroundColor Gray
        }
    
}

function Run-Setup {
    try {
        Write-Host "Running setup..." -ForegroundColor Blue
        start-process -wait "src\setup.bat"
 
    } catch {
        Write-Host "Error running setup: $_" -ForegroundColor Red
        Read-Host "Press Enter to continue..." | Out-Null
    }
}

function Run-Model {
    try {
        Write-Host "Model selection starting..." -ForegroundColor Blue
        start-process -wait "src\select_model.bat"
 
    } catch {
        Write-Host "Error running model selection: $_" -ForegroundColor Red
        Read-Host "Press Enter to continue..." | Out-Null
    }
}
$selection = ""
do {
    Show-Menu
    Write-Host "Please make a selection" -ForegroundColor Cyan -NoNewline
    $selection = Read-Host " "
    
    switch ($selection) {
        
        '1' { Run-Setup}
        '2' { Run-WithAI }
		'3' { Run-Alone }
		'4' { Run-Model }
        'q' { 
            Write-Host "Exiting..." -ForegroundColor Magenta
            return 
        }
    }
} until ($selection -eq 'q')
