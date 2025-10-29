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

<h3>Caption Options</h3>
<p><b>Caption Instruction:</b> Specific instructions for generating a detailed image caption.</p>
<p><b>Separate caption query:</b> Send a separate query just for captions. This will take twice as long and is of dubious value, but if you are using a model which has issues generating captions and keywords in one generation, you can split them up using this option.</p>
<p><b>Combined caption query:</b> Generate captions and keywords in one query (default).</p>
<p><b>No caption query:</b> Skip caption generation entirely, only create keywords. This option is a bit misleading because it will always generate a caption anyway, but this option will not write it to the metadata. Use this if you have captions you don't want to overwrite, but it won't make processing faster.</p>

<h3>Generation Options</h3>
<p><b>GenTokens:</b> Maximum number of tokens to generate in response. These are tokens, not words. Fewer tokens means faster processing per generation but may lead to more retries because the model may get cut off mid generation. More is not necessarily better though. Optimal range is between 150 and 300.</p>

<h3>Image Options</h3>
<p><b>Dimension length:</b> The maximum length of a horizontal or vertical dimension of the image, in pixels. Setting this higher will not necessarily result in better generations. Larger image sizes can take more memory and can lead to much slower processing. It is recommended to keep this between 392 and 896.</p> 

<h3>Sampler Options</h3>
<p><b>Temperature:</b> The randomness of the model output. Between 0.0 and 2.0</p>
<p><b>top_p:</b> Chooses from the smallest set of tokens which have a probability exceeding p. <i>Off = 1.0</i></p>
<p><b>top_k:</b> Limits to the most likely k tokens. <i>Off = 0</i></p>
<p><b>min_p:</b> Blocks tokens with probability lower than p. <i>Off = 0.0</i></p>
<p><b>rep_pen:</b> Prevents repetition. May cause erratic behavior. <i>Off = 1.0</i></p>

<h3>File Options</h3>
<p><b>Don't go in subdirectories:</b> Only process images in the main directory, don't look inside others.</p>
<p><b>Reprocess everything:</b> Process all images, even if they already have metadata. This will treat every image file as if it were brand new and the tool has never seen it before.</p>
<p><b>Reprocess failures:</b> Only reprocess images that failed in previous runs. Good idea to run with this option after a successful run to clean up stragglers.</p>
<p><b>Fix any orphans:</b> When a file gets processed it gets some metadata added to it so that the tool knows it has been processed and what the state of the last processing was. If we find images with what looks like valid metadata that was processed by the tool, but the status markers are missing, we call these orphans. This option will add the status marker to the orphans without regenerating the metadata. Without this checked then files which were produced with versions of the tool before the removal of the need for the json database will be processed again as new files. With this checked then if there is bad metadata in images that looks valid to the tool, it will mark those files as a success. It is recommended to use this option only if you have used previous versions of this tool before March 2025 and are running on those files again.</p>
<p><b>No backups:</b> Don't create backups of existing metadata before modifying. Exiftool will create a file with an _original label at the end as a backup for every one it alters. If you don't want one of those for every image processed by the tool, check this box.</p>
<p><b>Pretend mode:</b> Simulate processing without making any changes. This allows you to see what metadata would be generated without writing to any files.</p>
<p><b>No file validation:</b> Skip verifying file content. If you are seeing a lot of valid files being skipped as invalid, check this. Otherwise leave it alone.</p>
<p><b>No retries:</b> Don't retry failed API requests. This is when you don't want to bother trying a second time if you get a parse error from the AI. It is recommended to leave this disabled.</p>
<p><b>Use metadata sidecar instead of writing to image:</b> If you do not want to write anything to the image files themselves, for instance if you have hashed the files and they cannot change, you can instead write the metadata to an xmp file with the same name as the image file but with an xmp extension added. This xmp file will contain the metadata.</p>

<h3>Existing Metadata</h3>
<p><b>Don't clear existing keywords:</b> Keep existing keywords and add new ones. This adds the generated keywords to whatever keywords already exist in the image metadata. Very useful if you want to run the tool again on pictures with a different AI model and get some new keywords. Any existing keywords will be also processed according to the keyword corrections options below and deduplicated when combined with the new ones.</p>
<p><b>Don't clear existing caption:</b> Keep existing caption and add new text with tags. Checking this will add the generated caption onto an existing caption surrounded by XML tags.</p>

<h3>Keyword Corrections</h3>
<p>These options determine how the keywords are handled after the AI generates them. It is very important to process them properly because otherwise we end up with things like repetitions and nonsense. It is recommended to leave all of these at their defaults.
<p><b>Depluralize keywords:</b> Convert plural keywords to singular form.</p>
<p><b>Limit to N words:</b> Limit keywords to specified number of words each.</p>
<p><b>Split 'and'/'or' entries:</b> Break 'and'/'or' phrases into separate keywords unless they are in a list of exceptions like 'rock and roll'.</p>
<p><b>Ban prompt word repetitions:</b> AIs really like to repeat words back from the prompt if they aren't feeling very creative. Checking this option will refuse to add words that are in the instructions which commonly get repeated.</p>
<p><b>Cannot start with 3+ digits:</b> Filter out keywords starting with 3+ digits. '3d video' would be fine but '2024 summer' would be rejected.</p>
<p><b>Words must be 2+ characters:</b> Require words to be at least 2 characters long unless they are 'x' or 'u'.</p>
<p><b>Only Latin characters:</b> Remove keywords with non-Latin characters.</p>
"""

def get_settings_help():
    """Return the full settings help text"""
    return SETTINGS_HELP
