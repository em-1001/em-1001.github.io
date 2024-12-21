---
title:  "ServalShell"
excerpt: "NL to Bash AI project "

categories:
  - Natural Language Processing
tags:
  - Transformer
  - Project
last_modified_at: 2024-09-29T08:06:00-05:00
---


```
/content# ./servalshell.sh
Bashlint grammar set up (124 utilities)

                            
       \`*-.                    
        )  _`-.                 
       .  : `. .                
       : _   '  \               
       ; *` _.   `*-._          
       `-.-'          `-.       
         ;       `       `.     
         :.       .        \    
         . \  .   :   .-'   .   
         '  `+.;  ;  '      :   
         :  '  |    ;       ;-. 
         ; '   : :`-:     _.`* ;
      .*' /  .*' ; .*`- +'  `*' 
      `*-*   `*-*  `*-*'


ServalShell:~$ print current user name
translated bash: whoami
root

ServalShell:~$ ▯ 
```

# Paper: Attention Is All You Need

https://www.youtube.com/watch?v=bCz4OMemCcA

https://www.youtube.com/watch?v=AA621UofTUA&t=2s

# Project: NL to Bash translation AI with Transformer
## Model Test
```
ServalShell:~$ -h
If you enter a command in natural language, the program automatically translates it into a bash command and executes it.
If execution fails because the bash command translated by the model is incorrect, It will recommend several command structures.
Additionally, the following options are available.

-d [cmd],  --direct [cmd]              Execute bash command directly
-r [nl],  --recommend [nl]             Even if the command execution is successful, Recommended Command Structure is displayed
-h,  --help                            Describes usage and options
-q,  --quit                            Quit Servalshell

ServalShell:~$ print current user name
translated bash: whoami
root

ServalShell:~$ copies "file.txt" to "null.txt"
translated bash: cp "file.txt" "null.txt"

ServalShell:~$ prints "Hello, World!" on the terminal
translated bash: echo "Hello World!" | cat
Hello World!

ServalShell:~$ list current dictory files
translated bash: ls -l
total 1022116
-rw-r--r-- 1 root root      3485 Dec 21 14:58 beam_search.py
-rw-r--r-- 1 root root       870 Dec 21 14:58 config.py
drwxr-xr-x 4 root root      4096 Dec 21 14:58 Data
-rw-r--r-- 1 root root      3030 Dec 21 14:58 dataset.py
-rw-r--r-- 1 root root      7584 Dec 21 14:58 Description.md
-rw-r--r-- 1 root root        10 Dec 21 15:12 file.txt
-rw-r--r-- 1 root root      1064 Dec 21 14:58 LICENSE
-rw-r--r-- 1 root root      7692 Dec 21 14:58 model.py
-rw-r--r-- 1 root root 498658318 Mar 11  2024 nlc2bash-21epoch.zip
-rw-r--r-- 1 root root        10 Dec 21 15:13 null.txt
-rw-r--r-- 1 root root      1082 Dec 21 14:58 preprocess.py
drwxr-xr-x 2 root root      4096 Dec 21 15:00 __pycache__
-rw-r--r-- 1 root root      4229 Dec 21 14:58 README.md
-rw-r--r-- 1 root root       193 Dec 21 14:58 requirements.txt
-rw-r--r-- 1 root root       125 Dec 21 14:58 servalshell.sh
-rw-r--r-- 1 root root      4110 Dec 21 14:58 shell.py
drwxr-xr-x 5 root root      4096 Dec 21 14:58 Tellina
-rw-r--r-- 1 root root 547824092 Mar 11  2024 tellina21epoch.pth
-rw-r--r-- 1 root root     17028 Dec 21 15:11 tokenizer_cmd.json
-rw-r--r-- 1 root root     32693 Dec 21 15:11 tokenizer_invocation.json
-rw-r--r-- 1 root root     11276 Dec 21 14:58 train.py
-rw-r--r-- 1 root root      2800 Dec 21 14:58 translate.py

ServalShell:~$ Prints the current working directory
translated bash: pwd
/content/ServalShell

ServalShell:~$ gives execute permission to "script.sh"
translated bash: chmod Permission "script.sh"
chmod: invalid mode: ‘Permission’
Try 'chmod --help' for more information.

Recommended Command Structure
chmod Permission File
chmod Permission $( which Regex )
chmod +Permission File

ServalShell:~$ moves "file.txt" to "./bin"
translated bash: mv "file.txt" "./bin"

ServalShell:~$ creates a directory named "my_folder"
translated bash: mkdir "my_folder"

ServalShell:~$ changes to the "my_folder" directory
translated bash: cd "my_folder"

ServalShell:~$ displays the content of "flag.txt"
translated bash: cat "flag.txt"
SS{S3rv4ll_Sh3ll!!}
```


## Github Repository
[ServalShell](https://github.com/em-1001/ServalShell)
