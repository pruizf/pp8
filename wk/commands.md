/opt/storage/projects/o/pp7_llm/wk/disco$ for x in $(find -type d |grep "per-sonnet"|grep "txt") ; do echo $x ; done

/opt/storage/projects/o/pp7_llm/wk/per-sonnet-all$ mkdir ../per-sonnet-selected-random

/opt/storage/projects/o/pp7_llm/wk/per-sonnet-all$ for x in $(ls|shuf -n 25) ; do cp $x ../per-sonnet-selected-random/ ; done


