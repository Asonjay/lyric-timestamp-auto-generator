from aeneas.exacttiming import TimeValue
from aeneas.executetask import ExecuteTask
from aeneas.language import Language
from aeneas.syncmap import SyncMapFormat
from aeneas.task import Task
from aeneas.task import TaskConfiguration
from aeneas.textfile import TextFileFormat
import aeneas.globalconstants as gc
import json

""" For aeneas installation steps, visit https://github.com/readbeyond/aeneas """

# create Task object
config = TaskConfiguration()
config[gc.PPN_TASK_LANGUAGE] = Language.ENG
config[gc.PPN_TASK_IS_TEXT_FILE_FORMAT] = TextFileFormat.PLAIN
config[gc.PPN_TASK_OS_FILE_FORMAT] = SyncMapFormat.JSON
task = Task()
task.configuration = config
# configure to proper file paths for your computer
task.audio_file_path_absolute = "ABoyNamedSue.mp3"
task.text_file_path_absolute = "ABoyNamedSue.txt"
task.sync_map_file_path_absolute = "try.json"

# process Task
ExecuteTask(task).execute()

# output produced sync map  to json file
task.output_sync_map_file()

# open and inspect json file
f = open('try.json')

# returns JSON object as
# a dictionary
data = json.load(f)

# Iterating through the json
# list
for frag in data['fragments']:
    frag_end = frag['end']
    print(frag_end + ' ' + frag['lines'][0])

# Closing file
f.close()
