# #!/bin/bash

 
expect -c "
set timeout 10
spawn ssh -o StrictHostKeyChecking=no -X -l ${username} ${hostname} sh {hostfilename}.sh
expect \"${username}@{hostname}'s password: \"
send \"${password}\n\"
interact
"
