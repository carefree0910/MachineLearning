import os
import shutil
import platform

# Handle packages

try:
    from pynput.keyboard import Key, Listener, Controller
except ImportError:
    print("-- Warning: pynput is not installed on this machine, auto completion will not be supported --")
    print("-- You can type 'pip install pynput' to get pynput and then type 'refresh' to play with auto completion --")
    Key = Listener = Controller = None

# Handle cross-platform figures

_bad_slash, _tar_slash = "/", "\\"
if platform.system() != "Windows":
    _bad_slash, _tar_slash = "\\", "/"


class Util:

    @staticmethod
    def msg(dt, *args):
        if dt == "undefined_error":
            return Util.msg(
                "block_msg", "Error\n",
                "Undefined command '{}', type 'help' for more information\n".format(args[0]))
        if dt == "root_path_error":
            return Util.msg(
                "block_msg", "Error\n",
                "Current path '{}' is the root path\n".format(args[0]))
        if dt == "valid_path_error":
            return Util.msg(
                "block_msg", "Error\n",
                "'{}' is not a valid {}\n".format(args[0], args[1]))

        if dt == "block_msg":
            return "=" * 30 + "\n" + args[0] + "-" * 30 + "\n" + args[1] + "-" * 30
        if dt == "show_ls_message":
            title, folder_lst, file_lst = args
            body = (
                "\n".join([" folder - {}".format(_f) for _f in folder_lst]) + "\n" +
                "\n".join([" file   - {}".format(_f) for _f in file_lst]) + "\n"
            ) if file_lst or folder_lst else "-- Empty --\n"
            return Util.msg("block_msg", title, body)
        
    @staticmethod
    def get_cmd(msg):
        if Listener is not None:
            with Listener(on_press=CmdTool.on_press, on_release=CmdTool.on_release) as listener:
                cmd = input(msg).strip()
                listener.join()
                return cmd.replace(_bad_slash, _tar_slash)
        else:
            cmd = input(msg).strip()
            return cmd.replace(_bad_slash, _tar_slash)

    @staticmethod
    def get_formatted_error(err):
        return "-- {} --\n".format(err)

    @staticmethod
    def get_clean_path(path, user_platform):
        if user_platform == "Windows":
            while path.find("\\\\") >= 0:
                path = path.replace("\\\\", "\\")
        else:
            while path.find("//") >= 0:
                path = path.replace("//", "/")
        while len(path) > 3 and path[-1] == _tar_slash:
            path = path[:-1]
        return path

    @staticmethod
    def get_short_path(path, length):
        if len(path) >= length:
            return "..{}".format(path[len(path) - length:len(path)])
        return path

    @staticmethod
    def get_two_paths(cmd_tool, msgs, args):
        cmd = args[0].split()
        first_flag, second_flag = False, False
        first_arg, second_arg = "", ""
        if len(cmd) == 1:
            first_flag = second_flag = True
        elif len(cmd) == 2:
            first_arg = cmd[1]
            second_flag = True
        elif len(cmd) == 3:
            first_arg, second_arg = cmd[1:]
        if first_flag:
            first_arg = Util.get_cmd(msgs[0])
        if second_flag:
            second_arg = Util.get_cmd(msgs[1])
        first_path = cmd_tool.file_path if first_arg == "." else CmdTool.get_path(first_arg)
        second_path = cmd_tool.file_path if first_arg == "." else CmdTool.get_path(second_arg)
        return first_path, second_path

    @staticmethod
    def show_help_msg(dt):

        # Basic
        if dt == "help":
            print("Help (help)    -> type 'help' to see available commands,\n"
                  "                  type 'help **' to see the function of **")
        elif dt == "cd":
            print("Help (cd)      -> used to get into a folder")
        elif dt == "ls":
            print("Help (ls)      -> used to view files & folders")
        elif dt == "rm":
            print("Help (rm)      -> used to remove files or folders")
        elif dt == "mk":
            print("Help (mk)      -> used to make a file or a folder\n"
                  "                  it is recommended to type 'mk (dir) (type) (name)'\n"
                  "                  dir == '.' -> direct to current folder\n"
                  "                  assert(type in ('folder', 'file'))")
        elif dt == "mv":
            print("Help (mv)      -> used to move a file or a folder\n"
                  "                  it is recommended to type 'mv (old_dir) (new_dir)'\n")

        elif dt == "refresh":
            print("Help (refresh) -> used to refresh this command line tool.\n"
                  "                  you'll be sent back to 'Root' status")
        elif dt == "exit":
            print("Help (exit)    -> used to exit current status.\n"
                  "                  used to exit this command line tool when you're under 'Root' status")

        elif dt == "config":
            print("Help (config)  -> type 'config' to change configurations of all available tools,\n"
                  "                  type 'config rename' to change configurations of 'Rename' tool")

        # Root
        elif dt == "rename":
            print("Help (rename)  -> get use of 'Rename' tool")
        elif dt == "python":
            print("Help (python)  -> get use of Python")

        # Rename
        elif dt == "folder":
            print("Help (folder)  -> used to rename a folder in current folder\n"
                  "                  it is recommended to type 'folder (old_name) (new_name)'\n")
        elif dt == "file":
            print("Help (file)    -> used to rename a file in current folder\n"
                  "                  it is recommended to type 'file (old_name) (new_name)'\n")

        else:
            raise NotImplementedError

    @staticmethod
    def do_system_default(cmd):
        print(Util.msg("block_msg", "Caution\n", "Running default command in Windows command line\n"))
        os.system(cmd)


class CmdTool:

    _platform = platform.system()
    _file_path = os.getcwd()
    _self_path = "\"{}{}pycmd\".py".format(_file_path, _tar_slash)

    _current_command = []
    _do_auto_complete = True
    _auto_complete = ""
    _auto_complete_lst = []
    _auto_complete_track = []
    _auto_complete_flag = False
    _auto_complete_cursor = 0
    _auto_complete_finish = True

    if Controller is not None:
        _keyboard = Controller()
    else:
        _keyboard = None

    @classmethod
    def clear_cache(cls):
        cls._current_command = []
        cls._do_auto_complete = True
        cls._auto_complete = ""
        cls._auto_complete_lst = []
        cls._auto_complete_track = []

    @classmethod
    def click(cls, key):
        cls._keyboard.press(key)
        cls._keyboard.release(key)

    @classmethod
    def on_press(cls, key):
        if key == Key.tab and cls._current_command and cls._do_auto_complete:
            cls._auto_complete_flag = True
            cls._auto_complete_finish = False

            full_cmd = "".join(cls._current_command).split()
            cmd = full_cmd[0]
            if not cls._auto_complete_track:
                pth_head = full_cmd[-1]
                pth_head_len = len(pth_head)
            else:
                pth_head = "".join(cls._current_command).split(_tar_slash)[-1]
                pth_head_len = len(pth_head)

            track = _tar_slash.join(cls._auto_complete_track)
            cls._auto_complete_lst = [_f for _f in os.listdir(cls.get_path(track)) if _f[:pth_head_len] == pth_head]

            add_back_slash = False
            if cmd in ("cd", "ls"):
                cls._auto_complete_lst = [
                    _f for _f in cls._auto_complete_lst if os.path.isdir(cls.get_path(track + _tar_slash + _f))]
                add_back_slash = True
            if len(cls._auto_complete_lst) == 1:
                cls._auto_complete = cls._auto_complete_lst[0][pth_head_len:]
            else:
                cls._auto_complete = ""

            cls._current_command.append("\t")
            cls.click(Key.backspace)
            if cls._auto_complete:
                cls._auto_complete_track.append(pth_head + cls._auto_complete)
                if add_back_slash:
                    cls._auto_complete += _tar_slash
                cls._keyboard.type(cls._auto_complete)

            cls._auto_complete_flag = False
            if not cls._auto_complete:
                cls._auto_complete_finish = True

    @classmethod
    def on_release(cls, key):
        if not cls._auto_complete_flag:
            if key == Key.enter:
                cls.clear_cache()
                return False
            if key == Key.left or key == Key.right or key == Key.up:
                cls._do_auto_complete = False
            elif key == Key.backspace:
                if cls._auto_complete_finish:
                    cls._do_auto_complete = False
                if cls._current_command:
                    cls._current_command.pop()
            elif key == Key.space and cls._auto_complete_finish:
                cls._auto_complete_track = []
                cls._current_command.append(" ")
            elif key == Key.tab:
                pass
            else:
                if cls._auto_complete_finish:
                    char = str(key)[1:-1].strip()
                    if len(char) == 1:
                        cls._current_command.append(char)
                    elif char == "\\\\":
                        cls._current_command.append("\\")
                    elif char == "/":
                        cls._current_command.append("/")
                else:
                    if cls._auto_complete_cursor < len(cls._auto_complete):
                        cls._current_command.append(cls._auto_complete[cls._auto_complete_cursor])
                        cls._auto_complete_cursor += 1
                        if cls._auto_complete_cursor == len(cls._auto_complete):
                            cls._auto_complete_cursor = 0
                            cls._auto_complete_finish = True

    @classmethod
    def get_path(cls, ad):
        if not ad or ad == ".":
            return cls._file_path
        return cls._file_path + _tar_slash + ad
    
    @classmethod
    def get_path_from_cmd(cls, raw_cmd, cmd_type, tar_type, base_path=None, allow_no_param=True, custom_msg=""):
        
        if base_path is None:
            tmp_path = Util.get_clean_path(cls._file_path, CmdTool._platform)
        else:
            tmp_path = Util.get_clean_path(base_path, CmdTool._platform)

        if cmd_type and not allow_no_param:
            if raw_cmd == cmd_type:
                if not custom_msg:
                    raw_cmd = cmd_type + " " + Util.get_cmd("{} -> ".format(cmd_type))
                else:
                    raw_cmd = cmd_type + " " + Util.get_cmd(custom_msg)

            if raw_cmd[2] != " ":
                print(Util.msg("undefined_error", raw_cmd))
                return tmp_path

            if not raw_cmd or raw_cmd == ".":
                return tmp_path

        elif (not raw_cmd and not cmd_type) or (allow_no_param and raw_cmd == cmd_type):
            raw_cmd = "** ."

        if len(raw_cmd) > 3:
            raw_cmd = raw_cmd[3:]

        if raw_cmd[0] == ".":

            dot_counter = 1
            while dot_counter < len(raw_cmd) and raw_cmd[dot_counter] == ".":
                dot_counter += 1
            pth_length = tmp_path.count(_tar_slash) + 1
            if dot_counter > pth_length:
                print(Util.msg("block_msg", "Path Error\n",
                               "-- Too many '.': {} which exceed {} --".format(dot_counter, pth_length)))
                return tmp_path
            if dot_counter == len(raw_cmd):
                while dot_counter > 1:
                    tmp_path = tmp_path[:tmp_path.rfind(_tar_slash)]
                    dot_counter -= 1
            else:
                addition_path = raw_cmd[dot_counter:]
                while dot_counter > 1:
                    tmp_path = tmp_path[:tmp_path.rfind(_tar_slash)]
                    dot_counter -= 1
                tmp_path += _tar_slash + addition_path
                tmp_path = Util.get_clean_path(tmp_path, CmdTool._platform)

        else:
            tmp_path = Util.get_clean_path(CmdTool.get_path(raw_cmd), CmdTool._platform)

        if tar_type != "all" and (
            (tar_type.find("folder") >= 0 and not os.path.isdir(tmp_path)) or
            (tar_type.find("file") >= 0and not os.path.isfile(tmp_path))
        ):
            print(Util.msg("valid_path_error", tmp_path, tar_type))
            return tmp_path

        return tmp_path

    @property
    def file_path(self):
        return self._file_path

    def __init__(self, file_path=None):

        self._status = "root"
        if file_path is None:
            self._file_path = CmdTool._file_path
        else:
            self._file_path = CmdTool._file_path = file_path

        self._common_command = ("help", "config", "refresh", "exit")
        self._advance_command = ("cd", "ls", "rm", "mk", "mv", "cp")
        self._break_command = ("refresh", "exit")

    def _get_cmd(self, msg, format_path=False):

        def _cmd():
            if format_path:
                return Util.get_cmd(msg.format(self._file_path))
            return Util.get_cmd(msg)

        cmd = _cmd()
        while self._do_common_work(cmd):
            if self._status == "exit":
                return "exit"
            cmd = _cmd()
        return cmd

    def _cd(self, args):
        self._file_path = CmdTool.get_path_from_cmd(args[0], "cd", "folder", allow_no_param=False)
        self._update_path()
        return

    @classmethod
    def _ls(cls, args):
        pth = cls.get_path_from_cmd(args[0], "ls", "folder")

        folder_lst, file_lst = [], []
        for _f in os.listdir(pth):
            _p = pth + _tar_slash + _f
            if os.path.isdir(_p):
                folder_lst.append(_f)
            elif os.path.isfile(_p):
                file_lst.append(_f)

        try:
            print(Util.msg("show_ls_message", "Files & Folders in '{}':\n".format(pth), folder_lst, file_lst))
        except UnicodeEncodeError as err:
            print(Util.msg("block_msg", "Encoding Error\n", Util.get_formatted_error(err)))

    def _rm(self, args):
        pth = CmdTool.get_path_from_cmd(args[0], "rm", "all")

        if not os.path.isdir(pth):
            if not os.path.isfile(pth):
                print(Util.msg("block_msg", "Path Error\n", "-- '{}' not exists\n --".format(pth)))
            else:
                print(Util.msg("block_msg", "Removing\n", pth + "\n"))
                if self._get_cmd("(Root) Sure to proceed ? (y/n) -> ").lower() == "y":
                    try:
                        os.remove(pth)
                    except PermissionError as err:
                        print(Util.msg("block_msg", "Permission Error\n", Util.get_formatted_error(err)))
            return

        folder_lst, file_lst = [], []
        for _f in os.listdir(pth):
            _p = pth + _tar_slash + _f
            if os.path.isdir(_p):
                folder_lst.append(_f)
            elif os.path.isfile(_p):
                file_lst.append(_f)

        try:
            print(Util.msg("show_ls_message", "Removing '{}'\nWhich contains\n".format(pth), folder_lst, file_lst))
        except UnicodeEncodeError as err:
            print(Util.msg("block_msg", "Encoding Error\n", Util.get_formatted_error(err)))

        if self._get_cmd("(Root) Sure to proceed ? (y/n) -> ").lower() == "y":
            try:
                if os.path.isfile(pth):
                    os.remove(pth)
                else:
                    if pth == self._file_path:
                        self._do_common("cd", "cd ..")
                    shutil.rmtree(pth)
            except PermissionError as err:
                print(Util.msg("block_msg", "Permission Error\n", Util.get_formatted_error(err)))
        else:
            print("(Root) Caution -> Nothing happened")

    def _mk(self, args):

        mk = args[0].split()

        path_flag, type_flag, name_flag = False, False, False
        mk_path, mk_type, mk_name = "", "", ""

        if len(mk) == 1:
            path_flag = type_flag = name_flag = True
        elif len(mk) == 2:
            mk_path = mk[1]
            type_flag = name_flag = True
        elif len(mk) == 3:
            mk_path, mk_type = mk[1:]
            name_flag = True
        else:
            mk_path, mk_type = mk[1:3]
            mk_name = " ".join(mk[3:])

        if path_flag:
            mk_path = CmdTool.get_path(Util.get_cmd("(mk) dir -> "))
        else:
            mk_path = self._file_path if mk_path == "." else CmdTool.get_path(mk_path)
        if type_flag:
            mk_type = Util.get_cmd("(mk) type (folder or file) -> ")
        if name_flag:
            mk_name = Util.get_cmd("(mk) name -> ")

        if not os.path.isdir(mk_path):
            print(Util.msg("valid_path_error", mk_path, "folder"))
            return

        if not mk_type or mk_type not in ("folder", "file"):
            print("(Root) Caution -> Nothing happened")
            return

        mk_dir = mk_path + _tar_slash + mk_name

        if mk_type == "folder":
            try:
                os.mkdir(mk_dir)
            except FileExistsError as err:
                print(Util.msg("block_msg", "File Exists Error\n", Util.get_formatted_error(err)))
        elif mk_type == "file":
            if os.path.isfile(mk_dir):
                print(Util.msg("block_msg", "Error\n", "'{}' already exists\n".format(mk_dir)))
            else:
                try:
                    with open(mk_dir, "w"):
                        pass
                except PermissionError as err:
                    print(Util.msg("block_msg", "Permission Error\n", Util.get_formatted_error(err)))

    def _mv(self, args):
        old_path, new_path = Util.get_two_paths(
            self, ("(mv) Old dir -> ", "(mv) New dir -> "), args
        )

        if not os.path.isfile(old_path) and not os.path.isdir(old_path):
            print(Util.msg("valid_path_error", old_path, "dir"))

        try:
            shutil.move(old_path, new_path)
        except FileExistsError as err:
            print(Util.msg("block_msg", "File Exists Error\n", Util.get_formatted_error(err)))

    def _cp(self, args):
        old_path, new_path = Util.get_two_paths(
            self, ("(cp) Old dir -> ", "(cp) New dir -> "), args
        )

        if not os.path.isfile(old_path) and not os.path.isdir(old_path):
            print(Util.msg("valid_path_error", old_path, "dir"))

        try:
            if os.path.isfile(old_path):
                shutil.copyfile(old_path, new_path)
            else:
                shutil.copytree(old_path, new_path)
        except FileExistsError as err:
            print(Util.msg("block_msg", "File Exists Error\n", Util.get_formatted_error(err)))

    def _help(self, args):
        pass

    def _config(self, args):
        pass

    def _check_path(self, cmd):
        if len(cmd) >= 2 and cmd[1] == ":":
            if len(cmd) == 2:
                cmd += _tar_slash
            if os.path.isdir(cmd):
                self._file_path = cmd
                self._update_path()
                return True
            return False

    def _do_common(self, dt, *args):
        try:
            if dt in self._break_command:
                self._status = "exit"
                if dt == "refresh":
                    os.system("python {}".format(CmdTool._self_path))
                return False
            if dt == "help":
                self._help(args)
            elif dt == "config":
                self._config(args)
            elif dt == "cd":
                self._cd(args)
            elif dt == "ls":
                self._ls(args)
            elif dt == "rm":
                self._rm(args)
            elif dt == "mk":
                self._mk(args)
            elif dt == "mv":
                self._mv(args)
            elif dt == "cp":
                self._cp(args)
        except Exception as err:
            print(Util.msg("block_msg", "-- Special Error --\n", Util.get_formatted_error(err)))
        return True

    def _do_common_work(self, cmd):
        if self._status == "exit":
            return False
        if self._check_path(cmd):
            return True
        if cmd in self._common_command:
            return self._do_common(cmd)
        if cmd[:2] in self._advance_command:
            self._do_common(cmd[:2], cmd)
        elif cmd[:4] == "help":
            self._do_common("help", cmd[4:].strip())
        elif cmd[:6] == "config":
            self._do_common("config", cmd[6:].strip())
        else:
            return False
        return True

    def _renew_path(self, parent):
        parent._file_path = self._file_path

    def _update_path(self):
        os.chdir(self._file_path)
        CmdTool._file_path = self._file_path


class Rename(CmdTool):

    def __init__(self, file_path, pl, fl, whether_preview_detail, parent):

        CmdTool.__init__(self, file_path)
        self._parent = parent

        self._basic_batch_command = ("folders", "files")
        self._basic_command = ("folder", "file")
        self._special_command = ("batch", "seq")

        self._root_commands = [
            self._common_command, self._advance_command,
            self._basic_batch_command, self._basic_command,
            self._special_command
        ]
        self._batch_commands = [
            self._common_command, self._advance_command,
            self._basic_batch_command, self._basic_command
        ]

        self._batch_lst = []
        self._log_lst = []
        self._err_lst = []

        self._default_pl = pl
        self._default_fl = fl
        self._prev_detail = whether_preview_detail

        self._path_length = pl
        self._fn_length = fl
        self._fn_max = 10 ** self._fn_length - 1

    def _rename(self, args):
        old_path, new_path = Util.get_two_paths(
            self, (
                "{} (Rename {}) Old name -> ".format(args[1], args[0]),
                "{} (Rename {}) New name -> ".format(args[1], args[0])
            ), args
        )

        if (
            (args[0] == "file" and not os.path.isfile(old_path)) or
            (args[0] == "folder" and not os.path.isdir(old_path))
        ):
            print(Util.msg("valid_path_error", old_path, args[0]))
            return

        try:
            os.rename(old_path, new_path)
        except FileExistsError as err:
            print(Util.msg("block_msg", "File Exists Error\n", Util.get_formatted_error(err)))

    def _rename_batch(self, tar_type, tar_path=None, show_preview=True):

        tar_path = self._file_path if tar_path is None else tar_path

        if tar_type not in self._basic_batch_command:
            first_space = tar_type.find(" ")
            tar_type, addition_dir = tar_type[:first_space], tar_type[first_space + 1:]

        try:
            file_lst = os.listdir(tar_path)
        except FileNotFoundError as err:
            self._err_lst.append(Util.get_formatted_error(err))
            return

        counter = 0
        pipeline = []
    
        for i, file in enumerate(file_lst):
    
            old_dir = os.path.join(tar_path, file)

            if tar_type == "files" and os.path.isdir(old_dir):
                continue
            if tar_type == "folders" and os.path.isfile(old_dir):
                continue
    
            new_name = str(counter).zfill(self._fn_length)
            extension = os.path.splitext(file)[1]
            new_dir = os.path.join(tar_path, new_name + extension)

            pipeline.append((old_dir, new_dir))
            counter += 1

        if not pipeline:
            return

        flag = True

        if self._prev_detail and show_preview:
            self._preview_batch_detail(pipeline)
            _proceed = self._get_cmd("(Rename Batch) Sure to proceed ? (y/n) (Default: y) -> ")
            if _proceed and _proceed.lower() != "y":
                flag = False

        if not flag:
            return

        for old_dir, new_dir in pipeline:
            try:
                os.rename(old_dir, new_dir)
                self._log_lst.append("{} -> {}".format(
                    Util.get_short_path(old_dir, self._path_length),
                    Util.get_short_path(new_dir, self._path_length)
                ))
            except FileExistsError as err:
                self._err_lst.append(Util.get_formatted_error(err))
            
    def _do(self, dt, *args):
        if dt == "batch":
            self._batch()
            self._status = "root"
        elif dt == "finish_batch":
            self._finish_batch(args)
        elif dt == "handle_batch_result":
            self._handle_batch_result()
        elif dt == "seq":
            self._seq()
            self._status = "root"
        elif dt == "rename":
            self._rename(args)
        else:
            raise NotImplementedError

    def _help(self, args):

        if not args or not args[0]:
            if self._status == "root":
                rs = "\n".join([", ".join(_cmd) for _cmd in self._root_commands]) + "\n"
            elif self._status in self._special_command:
                rs = "\n".join([", ".join(_cmd) for _cmd in self._batch_commands]) + "\n"
            else:
                raise NotImplementedError
            print(Util.msg("block_msg", "Available commands:\n", rs))

        else:
            dt = args[0]

            if dt in self._common_command or dt in self._advance_command:
                Util.show_help_msg(dt)

            elif dt == "folders":
                print("Help (folders) -> used to rename folders in current folder.\n"
                      "                  By default, they will be like '{}', '{}' and so on".format(
                          "0".zfill(self._fn_length), "1".zfill(self._fn_length)))
            elif dt == "files":
                print("Help (files)   -> used to rename files in current folder.\n"
                      "                  They will be like '{}.***', '{}.***' and so on.\n"
                      "                  If you want to rename the files on your own, "
                      "please enter 'Batch' status".format(
                          "0".zfill(self._fn_length), "1".zfill(self._fn_length)))

            elif dt in self._basic_command:
                Util.show_help_msg(dt)

            elif dt == "batch":
                print("Help (batch)   -> only available under 'Rename Root' status, used to enter 'Batch' status.\n"
                      "                  You can build a 'pipeline' under 'Batch' status")
            elif dt == "seq":
                print("Help (seq)     -> only available under 'Rename Root' status, "
                      "used to enter 'Sequence' ('Seq') status.\n"
                      "                  You can use it to do specific sequential work. Useful only if:\n"
                      "                  1) you want to rename files or folders in folders naming '0000', '0001', ...\n"
                      "                  2) you want to rename them to "
                      "'{}.***', '{}.***', ... or "
                      "'{}', '{}', ...".format(
                          "0".zfill(self._fn_length), "1".zfill(self._fn_length),
                          "0".zfill(self._fn_length), "1".zfill(self._fn_length)
                      ))

            else:
                print("Help (error)   -> '{}' is not a valid command".format(dt))

    def _config(self, args):

        if not args or not args[0]:
            try:
                self._fn_length = int(Util.get_cmd("(Rename Config) (file_name_length)  -> "))
                self._path_length = int(Util.get_cmd("(Rename Config) (path_shown_length) -> "))
            except ValueError as err:
                print(Util.msg("block_msg", "Value Error\n", Util.get_formatted_error(err)))
                self._fn_length, self._path_length = self._default_fl, self._default_pl
            self._fn_max = 10 ** self._fn_length - 1

        else:
            dt = args[0]

            if dt == "fl":
                try:
                    self._fn_length = int(Util.get_cmd("(Rename Config) (file_name_length)  -> "))
                except ValueError as err:
                    print(Util.msg("block_msg", "Value Error\n", Util.get_formatted_error(err)))
                    self._fn_length = self._default_fl
                self._fn_max = 10 ** self._fn_length - 1

            elif dt == "pl":
                try:
                    self._path_length = int(Util.get_cmd("(Rename Config) (path_shown_length) -> "))
                except ValueError as err:
                    print(Util.msg("block_msg", "Value Error\n", Util.get_formatted_error(err)))
                    self._path_length = self._default_pl

    def _batch(self):

        self._status = "batch"

        while True:

            new_cmd = self._get_cmd("(Rename Batch) {} -> ", True)

            if self._status == "exit":
                break

            if new_cmd == "end":
                if self._batch_lst:
                    self._preview_batch()
                    self._do("finish_batch")
                break

            elif new_cmd in self._basic_batch_command:
                addition_dir = self._get_cmd("(Rename Batch) (Rename {}) dir -> ".format(new_cmd))
                tmp_path = CmdTool.get_path_from_cmd(addition_dir, "", "folder")
                if not os.path.isdir(tmp_path):
                    print(Util.msg("valid_path_error", tmp_path, "folder"))
                    continue
                self._batch_lst.append((new_cmd, tmp_path))

            elif new_cmd in self._basic_command:
                while True:
                    self._do("rename", new_cmd, "(Rename Batch)")
                    _continue = self._get_cmd("(Rename Batch) Continue ? (y/n) (default: y) -> ")
                    if _continue and _continue.lower() != "y":
                        break

            else:
                print(Util.msg("undefined_error", new_cmd))

    def _preview_batch_detail(self, pipeline):
        rs = "\n".join([
            Util.get_short_path(_old, self._path_length) + " -> " + Util.get_short_path(_new, self._path_length)
            for _old, _new in pipeline
        ]) + "\n"
        print(Util.msg("block_msg", "Batch Detail\n", rs))

    def _preview_batch(self):
        rs = "\n".join([
            Util.get_short_path(_p, self._path_length) + " -> {:>8s}".format(_c)
            for _c, _p in self._batch_lst
        ]) + "\n"
        print(Util.msg("block_msg", "Batch Preview\n", rs))

    def _finish_batch(self, args):
        args = True if not args else args[0]
        if len(self._batch_lst) != 0:
            for _c, _p in self._batch_lst:
                self._rename_batch(_c, _p, args)
        self._do("handle_batch_result")
        self._batch_lst = []

    def _handle_batch_result(self):
        if self._log_lst:
            print(Util.msg("block_msg", "Batch Results\n", "\n".join(self._log_lst) + "\n"))
            self._log_lst = []
        else:
            print(Util.msg("block_msg", "Batch Results\n", "None\n"))
        if self._err_lst:
            print(Util.msg("block_msg", "Batch Errors\n", "\n".join(self._err_lst) + "\n"))
            self._err_lst = []

    def _seq(self):

        self._status = "seq"

        while True:

            seq_cmd = self._get_cmd("(Rename Seq) {} -> ", True)

            if self._status == "exit":
                break

            if seq_cmd in self._basic_command:
                seq_cmd += "s"
            if seq_cmd not in self._basic_batch_command:
                print(Util.msg("undefined_error", seq_cmd))
                continue

            try:
                seq_start = int(self._get_cmd("(Rename Seq)   start point -> "))
                seq_end = int(self._get_cmd("(Rename Seq)   end point   -> "))
            except ValueError as err:
                print(Util.msg("block_msg", "Value Error\n", Util.get_formatted_error(err)))
                continue

            if seq_end < seq_start:
                print(Util.msg("block_msg", "Error\n", "start point '{}' exceeded end point '{}'\n".format(
                    seq_start, seq_end)))
                continue

            if seq_end >= self._fn_max:
                print(Util.msg("block_msg", "Error\n", "end point '{}' exceeded ceiling '{}'\n".format(
                    seq_end, self._fn_max)))
                continue

            name_lst = [
                Util.get_short_path(CmdTool.get_path(str(i).zfill(self._fn_length)), self._path_length)
                for i in range(seq_start, seq_end + 1)
            ]
            self._batch_lst = [
                (seq_cmd, CmdTool.get_path(str(i).zfill(self._fn_length)))
                for i in range(seq_start, seq_end + 1)
            ]
            print(Util.msg("block_msg", "Sequential Task ({})\n".format(seq_cmd), "\n".join(name_lst) + "\n"))

            _proceed = self._get_cmd("(Rename Seq) Sure to proceed ? (y/n) (Default: y) -> ")
            if _proceed and _proceed.lower() == "y":
                self._do("finish_batch", False)

    def _cmd_tool(self):

        while True:

            new_cmd = self._get_cmd("(Rename Root) {} -> ", True)

            if self._do_common_work(new_cmd):
                continue

            if self._status == "exit":
                self._renew_path(self._parent)
                break

            if new_cmd in self._special_command:
                self._do(new_cmd)

            elif new_cmd:
                cmd_lst = new_cmd.split()

                if cmd_lst:

                    if cmd_lst[0] in self._basic_batch_command:
                        self._rename_batch(new_cmd)
                        self._do("handle_batch_result")

                    elif cmd_lst[0] in self._basic_command:
                        self._do("rename", new_cmd, "(Rename Root)")

                else:
                    Util.do_system_default(new_cmd)

    def run(self, dt="cmd"):
        if dt == "cmd":
            self._cmd_tool()
        else:
            raise NotImplementedError


class PyCmd(CmdTool):

    def __init__(self):

        CmdTool.__init__(self)
        print("-- Welcome to pycmd, a light & extensible command line tool --")
        print("-- Your platform: {} --".format(CmdTool._platform))

        self._special_command = ("rename", )

        self._commands = [
            self._common_command, self._advance_command,
            self._special_command
        ]

        self._config = {
            "rename": (16, 4, True)
        }

    def _help(self, args):
        if not args or not args[0]:
            rs = "\n".join([", ".join(_cmd) for _cmd in self._commands]) + "\n"
            print(Util.msg("block_msg", "Available commands:\n", rs))

        else:
            dt = args[0]

            if dt in self._common_command or dt in self._advance_command or dt in self._special_command:
                Util.show_help_msg(dt)

            else:
                print("Help (error)   -> '{}' is not a valid command".format(dt))

    def _config(self, args):

        if not args or not args[0]:
            self._config_rename()

        else:
            dt = args[0]

            if dt == "rename":
                self._config_rename()

    def _config_rename(self):
        try:
            tmp_fl = int(Util.get_cmd("Config 'Rename' (file_name_length)  -> "))
            tmp_pl = int(Util.get_cmd("Config 'Rename' (path_shown_length) -> "))
            tmp_pd = bool(Util.get_cmd("Config 'Rename' (preview_batch_detail) -> "))
            self._config["rename"] = (tmp_fl, tmp_pl, tmp_pd)
        except ValueError as err:
            print(Util.msg("block_msg", "Value Error\n", Util.get_formatted_error(err)))

    def _do_special(self, cmd):

        self._status = cmd

        if cmd == "rename":
            pl, fl, pd = self._config["rename"]
            rename = Rename(self._file_path, pl, fl, pd, self)
            rename.run()

    def run(self):

        while True:

            new_cmd = self._get_cmd("(Root) {} -> ", True)

            if self._do_common_work(new_cmd):
                continue

            if self._status == "exit":
                break

            if new_cmd in self._special_command:
                self._do_special(new_cmd)

            elif new_cmd:
                Util.do_system_default(new_cmd)


if __name__ == "__main__":
    tool = PyCmd()
    tool.run()
