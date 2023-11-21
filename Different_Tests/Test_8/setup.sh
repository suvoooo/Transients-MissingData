# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/lhome/ext/uv111/uv1111/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/lhome/ext/uv111/uv1111/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/lhome/ext/uv111/uv1111/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/lhome/ext/uv111/uv1111/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate /lhome/ext/uv111/uv1111/anaconda3/envs/pytorch_venv
