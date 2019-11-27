# #!/bin/bash

expect -c "
set timeout 10
spawn scp -r ./shell/received.bmp ${username}@${hostname}:${path}
expect \"${username}@${hostname}'s password: \"
send \"${pass}\n\"
interact
"
