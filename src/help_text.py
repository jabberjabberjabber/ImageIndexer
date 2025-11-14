"""
Help text content for the Image Indexer GUI settings
"""

SETTINGS_HELP = """
<h2>Settings Help</h2>

<h3>API Settings</h3>
<p><b>API URL:</b> URL of the LLM API server. Default is http://localhost:5001</p>
<p><b>API Password:</b> Password for API authentication if required. Leave blank if no authentication needed.</p>

<h3>Instruction Settings</h3>
<p><b>System Instruction:</b> The instruction given to guide the LLM's behavior.</p>
<p><b>Edit Instruction:</b> Opens dialog to edit detailed instructions for image analysis.</p>

<h3>Directory Setting</h3>
<p><b>Skip Folders:</b> list folders here that you want to skip entirely. Separate by a new line or a semicolon. You don't have to put the full path -- the subdirectory name only will suffice.</p>

<h3>Caption Options</h3>
<p><b>Caption Instruction:</b> Specific instructions for generating a detailed image caption.</p>
<p><b>Separate caption query:</b> Send a separate query for captions and keywords This will take twice as long. Uses the Caption instruction and Keyword instruction.</p>
<p><b>Combined caption query:</b> Generate captions and keywords in one query. This uses the main instruction. <i>Recommended setting</i></p>
<p><b>No caption query:</b> Skip caption generation entirely, only create keywords. This uses the Keyword instruction.</p>

<h3>Generation Options</h3>
<p><b>GenTokens:</b> Maximum number of tokens to generate in response. These are tokens, not words. Fewer tokens means faster processing per generation but may lead to more retries because the model may get cut off mid generation. More is not necessarily better though. <i>Recommended setting for separate captions or keywords is between 100 and 200, for combined caption and keywords between 200 and 300</i></p>

<h3>Image Options</h3>
<p><b>Dimension length:</b> The maximum length of a horizontal or vertical dimension of the image, in pixels. Setting this higher will not necessarily result in better generations. <i>Recommended setting is between 392 and 896</i></p>

<h3>Sampler Options</h3>
<p>Samplers affect the tokens that the AI can choose from every time it generates a new token.</p>
<p><b>Temperature:</b> The randomness of the model output. Between 0.0 and 2.0 <i>Recommended setting is between 0.1 and 0.5</i></p>
<p><b>top_p:</b> Chooses from the smallest set of tokens which have a probability exceeding p. Off = 1.0 <i>Recommended setting is between 0.92 and 1</i></p>
<p><b>top_k:</b> Limits to the most likely k tokens. Off = 0 <i>Recommended setting is between 20 and 100</i></p>
<p><b>min_p:</b> Blocks tokens with probability lower than p. Off = 0.0 <i>Recommended setting is between 0.02 and 0.05</i></p>
<p><b>rep_pen:</b> Prevents repetition. May cause erratic behavior. Off = 1.0 <i>Recommended setting is between 1.0 and 1.02</i></p>

<h3>File Options</h3>
<p><b>Don't go in subdirectories:</b> Only process images in the main directory, don't look inside others.</p>
<p><b>Reprocess everything:</b> Process all images, even if they already have metadata. If you check this and leave <i>Don't clear existing keywords</i> unchecked it will remove all existing keywords from any previously processed files and replace them with the new generations.</p>
<p><b>Reprocess failures:</b> Reprocess images that were marked as failed in previous runs.</p>
<p><b>Fix any orphans:</b> <i>Deprecated setting</i> When a file gets processed it gets some metadata added to it so that the tool knows it has been processed and what the state of the last processing was. If we find images with what looks like valid metadata that was processed by the tool, but the status markers are missing, we call these orphans. This option will add the status marker to the orphans without regenerating the metadata. Without this checked then files which were produced with versions of the tool before the removal of the need for the json database will be processed again as new files. With this checked then if there is bad metadata in images that looks valid to the tool, it will mark those files as a success. It is recommended to use this option only if you have used previous versions of this tool before March 2025 and are running on those files again.</p>
<p><b>No backups:</b> Don't create backups of existing metadata before modifying. By default there will be a file created with an <i>_original</i> label for each altered image, so if you don't want that then check this box.</p>
<p><b>Pretend mode:</b> Simulate processing without making any changes.</p>
<p><b>No file validation:</b> If checked, files will not be checked for metadata errors.</p>
<p><b>No retries:</b> If a generation fails a second attempt will occur unless this box is checked.</p>
<p><b>Use metadata sidecar instead of writing to image:</b> If you do not want to write anything to the image files themselves, for instance if you have hashed the files and they cannot change, you can instead write the metadata to an xmp file with the same name as the image file but with an xmp extension added. This xmp file will contain the metadata.</p>

<h3>Existing Metadata</h3>
<p><b>Don't clear existing keywords:</b> Checking this box will add the newly generated keywords to any existing keywords in the metadata. This option is useful for running the generations again with a new model or a new prompt along with <i>Reprocess all</i> to add try and fill them out. Will not add duplicates.</p>
<p><b>Don't clear existing caption:</b> Keep existing caption and add new text with tags. Checking this will add the generated caption onto an existing caption surrounded by XML tags.</p>

<h3>Keyword Corrections</h3>
<p>These options determine how the keywords are handled after the AI generates them.</p>
<p><b>Depluralize keywords:</b> Convert plural keywords to singular form. May strip the last <i>s</i> off the end of some words.</p>
<p><b>Limit to N words:</b> Limit keywords to specified number of words each.</p>
<p><b>Split 'and'/'or' entries:</b> Break <i>and</i> / <i>or</i> phrases into separate keywords unless they are in a list of exceptions like 'rock and roll'.</p>
<p><b>Ban prompt word repetitions:</b> AIs really like to repeat words back from the prompt if they aren't feeling very creative. Checking this option will refuse to add words that are in the instructions which commonly get repeated.</p>
<p><b>Cannot start with 3+ digits:</b> Filter out keywords starting with 3+ digits. <i>3d video</i> would be fine but <i>2024 summer</i> would be rejected.</p>
<p><b>Words must be 2+ characters:</b> Require words to be at least 2 characters long unless they are <i>x</i> or <i>u</i>.</p>
<p><b>Only Latin characters:</b> Remove keywords with non-Latin characters.</p>
"""

def get_settings_help():
    """Return the full settings help text"""
    return SETTINGS_HELP
