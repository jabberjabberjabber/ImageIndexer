#!/bin/bash

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
GRAY='\033[0;37m'
WHITE='\033[1;37m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESOURCES_DIR="$SCRIPT_DIR/../resources"
KOBOLD_ARGS_PATH="$RESOURCES_DIR/kobold_args.json"

show_menu() {
    clear
    echo -e "${CYAN}================ Indexer Launcher ================${NC}"
    echo ""
    echo -e "${YELLOW}1:${NC} ${GREEN}Install Requirements${NC}"
    echo -e "${YELLOW}2:${NC} ${GREEN}Run Indexer with Model${NC}"
    echo -e "${YELLOW}3:${NC} ${GREEN}Run Indexer Alone${NC}"
    echo -e "${YELLOW}4:${NC} ${GREEN}Select Model${NC}"
    echo -e "${YELLOW}Q:${NC} ${RED}Quit${NC}"
    echo ""
}

run_with_ai() {
    if [ -f "$KOBOLD_ARGS_PATH" ]; then
        echo -e "${CYAN}Loading configuration...${NC}"
        
        if command -v jq >/dev/null 2>&1; then
            EXECUTABLE=$(jq -r '.executable' "$KOBOLD_ARGS_PATH")
            MODEL_PARAM=$(jq -r '.model_param' "$KOBOLD_ARGS_PATH")
            MMPROJ=$(jq -r '.mmproj' "$KOBOLD_ARGS_PATH")
            CONTEXTSIZE=$(jq -r '.contextsize' "$KOBOLD_ARGS_PATH")
            VISIONMAXRES=$(jq -r '.visionmaxres' "$KOBOLD_ARGS_PATH")
			CHATCOMPLETIONSADAPTER=$(jq -r '.chatcompletionsadapter' "$KOBOLD_ARGS_PATH")
			FLASHATTENTION=$(jq -r '.flashattention' "$KOBOLD_ARGS_PATH")
        else
            EXECUTABLE=$(python3 -c "import json; print(json.load(open('$KOBOLD_ARGS_PATH'))['executable'])")
            MODEL_PARAM=$(python3 -c "import json; print(json.load(open('$KOBOLD_ARGS_PATH'))['model_param'])")
            MMPROJ=$(python3 -c "import json; print(json.load(open('$KOBOLD_ARGS_PATH'))['mmproj'])")
            CONTEXTSIZE=$(python3 -c "import json; print(json.load(open('$KOBOLD_ARGS_PATH'))['contextsize'])")
            VISIONMAXRES=$(python3 -c "import json; print(json.load(open('$KOBOLD_ARGS_PATH'))['visionmaxres'])")
			CHATCOMPLETIONSADAPTER=$(python3 -c "import json; print(json.load(open('$KOBOLD_ARGS_PATH'))['chatcompletionsadapter'])")
			FLASHATTENTION=$(python3 -c "import json; print(json.load(open('$KOBOLD_ARGS_PATH'))['flashattention'])")
		fi

        EXECUTABLE_PATH="$RESOURCES_DIR/$EXECUTABLE"

        echo -e "${GREEN}Starting Indexer with AI support...${NC}"
        echo -e "${GRAY}Executable:${NC} ${WHITE}$EXECUTABLE_PATH${NC}"

        # Set the working directory to the koboldcpp exec directory
        WORKING_DIR=$(dirname "$EXECUTABLE_PATH")

        chmod +x "$EXECUTABLE_PATH"

        # Build command with conditional flashattention flag
        FLASHATTENTION_FLAG=""
        if [ "$FLASHATTENTION" = "true" ] || [ "$FLASHATTENTION" = "True" ]; then
            FLASHATTENTION_FLAG="--flashattention"
        fi

        # Start the process in the background
        (cd "$WORKING_DIR" && "$EXECUTABLE_PATH" "$MODEL_PARAM" --mmproj "$MMPROJ" $FLASHATTENTION_FLAG --contextsize "$CONTEXTSIZE" --visionmaxres "$VISIONMAXRES" --chatcompletionsadapter "$CHATCOMPLETIONSADAPTER") &
        
        run_gui
    else
        echo -e "${RED}Error: Kobold arguments file not found at $KOBOLD_ARGS_PATH${NC}"
        read -p "Press Enter to continue..."
    fi
}

run_alone() {
    echo -e "${GREEN}Running Indexer alone...${NC}"
    run_gui
}

run_gui() {
    echo -e "${CYAN}Launching Python GUI component...${NC}"
    
    chmod +x "$SCRIPT_DIR/gui.sh"
    
    bash "$SCRIPT_DIR/gui.sh"
}

run_setup() {
    echo -e "${BLUE}Running setup...${NC}"
    
    chmod +x "$SCRIPT_DIR/setup.sh"
    
    bash "$SCRIPT_DIR/setup.sh"
}
run_model() {
    echo -e "${BLUE}Model selection starting...${NC}"
    
    chmod +x "$SCRIPT_DIR/setup.sh"
    
    bash "$SCRIPT_DIR/setup.sh"
}
selection=""
while [ "$selection" != "q" ]; do
    show_menu
    echo -ne "${CYAN}Please make a selection${NC} "
    read selection
    
    case $selection in
        1) run_setup ;;
		2) run_with_ai ;;
        3) run_alone ;;
        4) run_model ;;
        q|Q) 
            echo -e "${MAGENTA}Exiting...${NC}"
            exit 0
            ;;
        *) echo -e "${RED}Invalid selection${NC}" ;;
    esac
done
