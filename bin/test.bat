@echo off

for /L %%i in (5,1,49) do (
    python bin/test.py --snapshot "models/snapshot/checkpoint_e%%i.pth"
)


@REM python siamese_tracking/train_siamfc.py