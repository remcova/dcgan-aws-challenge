def is_local_master(use_hvd, horovod):
    return use_hvd is False or horovod.local_rank() == 0


def is_not_worker_zero(use_hvd, horovod):
    if use_hvd is True and horovod.rank() != 0:
        # not worker zero
        return False
    elif use_hvd is True and horovod.rank() == 0:
        # master
        return True
    elif use_hvd is False:
        return False
