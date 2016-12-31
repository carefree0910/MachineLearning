# coding: utf-8
import math
import time
import pickle
import threading
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as tkm

import Gen


class TravelGUI(tk.Frame):

    def __init__(self):

        # ==============================================================================================================
        # Initialize Frame

        tk.Frame.__init__(self)
        self.master.title("Travel Helper")
        self.master.protocol("WM_DELETE_WINDOW", lambda: None)
        self.grid()

        self.all_frame = tk.LabelFrame()
        self.route_frame = tk.LabelFrame(self.all_frame)
        self.info_frame = tk.LabelFrame(self.all_frame)
        self.sug_frame = tk.LabelFrame(self.all_frame)

        self.all_frame.grid()
        self.route_frame.grid(row=0, column=1, rowspan=2)
        self.info_frame.grid(row=3, column=1, columnspan=3)
        self.sug_frame.grid(row=4, column=1)

        # ==============================================================================================================
        # Initialize Status

        self.admin = False
        self.special_admin = False

        # ==============================================================================================================
        # Initialize Pictures

        self.logo_pic = tk.PhotoImage(file="Pictures/logo.png")
        self.search_pic = tk.PhotoImage(file="Pictures/search.png")
        self.save_pic = tk.PhotoImage(file="Pictures/save.png")
        self.add_pic = tk.PhotoImage(file="Pictures/add.png")
        self.del_pic = tk.PhotoImage(file="Pictures/del.png")
        self.confirm_pic = tk.PhotoImage(file="Pictures/confirm.png")
        self.publisher_pic = tk.PhotoImage(file="Pictures/publisher.png")

        # ==============================================================================================================
        # Initialize Animations

        self.cast = False
        self.cast_count = -1
        self.cast_lst = [tk.PhotoImage(file="Pictures/Animation/Cast/cast0{}.png".format(i + 1)) for i in range(6)]
        cast_temp = [tk.PhotoImage(file="Pictures/Animation/Cast/cast{}.png".format(i + 1)) for i in range(66)]
        self.cast_lst += cast_temp
        self.cast_thread = threading.Thread(target=self.cast_begin)

        self.famous = False
        self.famous_count = -1
        behind = ["00{}".format(i) for i in range(1, 10)]
        behind += ["0{}".format(i) for i in range(10, 15)]
        self.famous_lst = []
        for i in range(1, 8):
            behind.reverse()
            self.famous_lst += [tk.PhotoImage(file="Pictures/Famous/{}11_{}.png"
                                              .format(i, j))for j in behind]
            behind.reverse()
            self.famous_lst += [tk.PhotoImage(file="Pictures/Famous/{}11_{}.png"
                                              .format(i, j))for j in behind]
        self.famous_name_lst = ["八达岭长城", "北京大学", "故宫", "国家大剧院", "石花洞", "南锣鼓巷", "Easter"]
        self.famous_thread = threading.Thread(target=self.famous_begin)

        # ==============================================================================================================
        # Initialize Initial Point

        self.init_frame = tk.LabelFrame(self.route_frame, text="Search & Choose your departure spot")
        self.init_entry = tk.Entry(self.init_frame, width=26)
        self.init_entry.bind("<KeyRelease>", lambda event: self.refresh_init())
        self.init_entry.bind("<Return>", lambda event: self.set_init_spot())
        self.init_entry.insert(0, "北京大学")
        self.init_expand1 = tk.Label(self.init_frame, width=2)
        self.init_expand2 = tk.Label(self.init_frame, width=1)
        self.init_expand3 = tk.Label(self.init_frame, width=1)
        self.init_expand4 = tk.Label(self.init_frame, width=1)
        self.init_lst = []
        self.init_choice_box = ttk.Combobox(self.init_frame, value=self.init_lst, width=40, state="readonly")
        self.init_choice_box.set("北京大学")
        self.init_btn = tk.Button(self.init_frame, image=self.confirm_pic, bd=0, command=self.set_init_spot)

        self.init_entry.grid(row=0, column=1)
        self.init_choice_box.grid(row=0, column=3)
        self.init_btn.grid(row=0, column=5)
        self.init_expand1.grid(row=0, column=0)
        self.init_expand2.grid(row=0, column=2)
        self.init_expand3.grid(row=0, column=4)
        self.init_expand4.grid(row=0, column=6)

        self.init_frame.grid(row=2, column=1, columnspan=2)
        self.departure = "北京大学"

        # ==============================================================================================================
        # Initialize Search Route

        self.targets = set()

        self.search_route_frame = tk.LabelFrame(self.route_frame, text="Search for routes")
        self.search_route_frame.grid(row=0, column=1)

        self.search_expand1 = tk.Label(self.search_route_frame, width=9)
        self.search_expand2 = tk.Label(self.search_route_frame, width=9)
        self.search_str = tk.StringVar()
        self.search_str.set("Target not selected yet")
        self.search_label = tk.Label(self.search_route_frame, textvariable=self.search_str)
        self.search = tk.Entry(self.search_route_frame, width=40)
        self.search_btn = tk.Button(self.search_route_frame, image=self.search_pic,
                                    command=self.search_route, bd=0)
        self.search_add = tk.Button(self.search_route_frame, image=self.add_pic,
                                    command=self.add_search_spot, bd=0)
        self.search.bind("<Return>", lambda event: self.add_search_spot())
        self.search_del = tk.Button(self.search_route_frame, image=self.del_pic,
                                    command=self.del_search_spot, bd=0)
        self.search_label.grid(row=0, columnspan=15)
        self.search.grid(row=1, column=1, columnspan=10)
        self.search_add.grid(row=1, column=11)
        self.search_del.grid(row=1, column=12)
        self.search_expand1.grid(row=1, column=0)
        self.search_expand2.grid(row=1, column=14)

        # ==============================================================================================================
        # Initialize Choose Priority

        self.choose_priority_frame = tk.LabelFrame(self.route_frame, text="Choose tendency")
        self.choose_priority_frame.grid(row=1, column=1)

        self.search_expand3 = tk.Label(self.choose_priority_frame, width=3)
        self.search_expand3.grid(row=1, column=0)
        self.search_expand4 = tk.Label(self.choose_priority_frame, width=3)
        self.search_expand4.grid(row=1, column=2)
        self.search_expand5 = tk.Label(self.choose_priority_frame, width=3)
        self.search_expand5.grid(row=1, column=4)

        self.prior = tk.StringVar()
        self.prior.set("less time")

        self.cost_prior = tk.Button(self.choose_priority_frame, width=12, text="less cost",
                                    command=lambda: self.prior.set("less cost"), bd=0, bg="PaleTurquoise")
        self.time_prior = tk.Button(self.choose_priority_frame, width=12, text="less time",
                                    command=lambda: self.prior.set("less time"), bd=0, bg="PaleTurquoise")
        self.change_prior = tk.Button(self.choose_priority_frame, width=12, text="less change",
                                      command=lambda: self.prior.set("less change"), bd=0, bg="PaleTurquoise")

        self.cost_prior.grid(row=1, column=1)
        self.time_prior.grid(row=1, column=3)
        self.change_prior.grid(row=1, column=5)

        self.prior_status = tk.Label(self.choose_priority_frame, textvariable=self.prior, width=21)
        self.prior_status.grid(row=1, column=6)

        # ==============================================================================================================
        # Initialize Information

        self.names, self.labels, self.label_set, self.locs, self.loc_set, self.vectors, self.total_vector = \
            [None] * 7

        def sub_init():
            time.sleep(1.5)
            self.names, self.labels, self.label_set, self.locs, self.loc_set, \
                self.vectors, self.total_vector = Gen.get_info()

        class MainThread:

            @staticmethod
            def call_func(top):
                thread = threading.Thread(target=sub_init)
                thread.start()
                btn = tk.Label(top, bd=0, bg="Silver", text="Initializing...", width=8)
                btn.grid(row=2, columnspan=20)
                count = 0
                while thread.is_alive():
                    count += 1
                    if count < 12:
                        btn["width"] += 4
                    top.update()
                    time.sleep(0.2)
                btn.destroy()
                self.master.protocol("WM_DELETE_WINDOW", self.close_gui)

        MainThread.call_func(self.all_frame)
        self.search_btn.grid(row=1, column=13)
        self.search.focus_set()

        self.logo = tk.Button(self.all_frame, image=self.logo_pic, width=100, bd=0, command=self.cast_begin)
        self.logo.grid(row=0, column=0)
        self.publisher = tk.Label(self.all_frame, image=self.publisher_pic)
        self.publisher.grid(row=4, column=3)
        self.cast_show = tk.Label(self.all_frame)
        self.cast_show.grid(row=1, column=0, rowspan=3)
        self.famous_btn = tk.Button(self.all_frame, bd=0)
        self.famous_btn.grid(row=1, column=3)

        # ==============================================================================================================
        # Initialize Administrator

        self.admin_frame = tk.LabelFrame(self.all_frame, text="Administrator")
        self.admin_frame.grid(row=0, column=3)

        self.password = ""
        self.special_password = "carefree0910"
        self.admin_str = tk.StringVar()
        self.admin_str.set("Enter")
        self.admin_btn = tk.Button(self.admin_frame, command=self.create_admin_top,
                                   textvariable=self.admin_str, width=16, bd=0, bg="Lavender")
        self.admin_save = tk.Button(self.admin_frame, text="save", width=16,
                                    bd=0, bg="MintCream", command=self.save)
        self.admin_btn.grid(row=0, column=0)
        self.admin_cursor = 0

        # ==============================================================================================================
        # Initialize Search Information

        self.search_info_frame = tk.LabelFrame(self.info_frame, text="Search for information")
        self.search_info_frame.grid(row=0, column=1, columnspan=3)

        self.search_expand7 = tk.Label(self.search_info_frame, width=3)
        self.search_expand7.grid(row=0, column=0)
        self.search_expand8 = tk.Label(self.search_info_frame, width=3)
        self.search_expand8.grid(row=0, column=2)
        self.search_expand9 = tk.Label(self.search_info_frame, width=3)
        self.search_expand9.grid(row=0, column=4)
        self.search_expand10 = tk.Label(self.search_info_frame, width=3)
        self.search_expand10.grid(row=0, column=6)
        self.search_expand11 = tk.Label(self.search_info_frame, width=3)
        self.search_expand11.grid(row=0, column=8)

        self.search_name = tk.Entry(self.search_info_frame, width=34)
        self.search_name.insert(0, "Name")
        self.search_name.bind("<Return>", self.search_info())
        self.search_type = ttk.Combobox(self.search_info_frame, width=14, state="readonly",
                                        values=self.labels)
        self.search_type.set("Type")
        self.search_loc = ttk.Combobox(self.search_info_frame, width=10, state="readonly",
                                       values=self.locs)
        self.search_loc.set("Location")

        self.search_name.grid(row=0, column=1)
        self.search_type.grid(row=0, column=3)
        self.search_loc.grid(row=0, column=5)

        self.confirm_btn = tk.Button(self.search_info_frame, text="   Get Information!   ",
                                     command=self.search_info(), bd=0, bg="Thistle")
        self.confirm_btn.grid(row=0, column=7)

        # ==============================================================================================================
        # Initialize Suggestions

        self.show_sug_frame = tk.LabelFrame(self.sug_frame, text="Spots you might be interested in")
        self.show_sug_frame.grid()

        self.show_sug_lst = ["北京大学"]
        self.show_sug_box = ttk.Combobox(self.show_sug_frame, value=self.show_sug_lst, width=52, state="readonly")
        self.show_sug_box.set(self.show_sug_lst[0])
        self.show_sug_button = tk.Button(self.show_sug_frame, text="Go!", bd=0, bg="Ivory", width=12,
                                         command=self.switch_to_route(self.show_sug_box))

        self.sug_expand1 = tk.Label(self.show_sug_frame, width=3)
        self.sug_expand2 = tk.Label(self.show_sug_frame, width=3)
        self.sug_expand3 = tk.Label(self.show_sug_frame, width=3)

        self.show_sug_box.grid(row=0, column=1)
        self.show_sug_button.grid(row=0, column=3)
        self.sug_expand1.grid(row=0, column=0)
        self.sug_expand2.grid(row=0, column=2)
        self.sug_expand3.grid(row=0, column=4)
        self.show_sug = False
        self.check_vector()

        # ==============================================================================================================
        # Initialize Menu

        self.menu = tk.Menu()
        self.menu.add_command(label="Help", command=TravelGUI.show_help(self.search))
        self.master["menu"] = self.menu

        # ==============================================================================================================
        # Initialize Shortcuts

        TravelGUI.bind_iconify(self.master, self.master)
        self.master.bind("<Escape>", lambda event: self.close_gui())
        TravelGUI.bind_help(self.master, self.search)
        self.master.bind("<Alt-q>", lambda event: self.search_route())
        self.master.bind("<Alt-w>", lambda event: self.search_str.set("Target not selected yet"))
        self.master.bind("<Control-g>", TravelGUI.show_info(self.search))
        self.master.bind("<Control-p>", lambda event: self.create_admin_top())
        self.master.bind("<Control-s>", lambda event: self.save())
        self.master.bind("<Control-i>", lambda event: TravelGUI.focus_and_select(self.search_name))
        self.master.bind("<Control-r>", lambda event: TravelGUI.focus_and_select(self.search))
        self.master.bind("<Control-q>", lambda event: TravelGUI.focus_and_select(self.init_entry))
        self.master.bind("<Control-Alt-a>", lambda event: self.prior.set("less cost"))
        self.master.bind("<Control-Alt-s>", lambda event: self.prior.set("less time"))
        self.master.bind("<Control-Alt-d>", lambda event: self.prior.set("less change"))

        # ==============================================================================================================
        # Begin !!!!!!

        self.search.focus_set()
        self.ani_begin()

    # ------------------------------------------------------------------------------------------------------------------
    # Animations

    def ani_begin(self):
        self.cast_thread.start()
        self.famous_thread.start()

    def cast_begin(self):
        self.cast_count = -1
        self.cast_show["image"] = self.cast_lst[0]
        self.cast = True if not self.cast else False
        while self.cast:
            self.cast_count += 1
            pic = self.cast_lst[0]
            if 4 < self.cast_count < 77:
                pic = self.cast_lst[self.cast_count - 5]
            self.cast_show["image"] = pic
            self.update()
            time.sleep(0.07)
            if self.cast_count > 80:
                self.cast_count = -1

    def famous_begin(self):
        self.famous_count = -1
        self.famous_btn["image"] = self.famous_lst[0]
        self.famous = True if not self.famous else False
        while self.famous:
            self.famous_count += 1
            pic, name = self.famous_lst[self.famous_count], self.famous_name_lst[self.famous_count // 28]
            self.famous_btn["image"] = pic
            if name == "Easter":
                self.famous_btn["command"] = lambda: tkm.showinfo("Easter Egg!",
                                                                  "-> 前排广告位出租(￣△￣；) \n"
                                                                  "-> 租金：一个赞(￣▽￣')")
            else:
                self.famous_btn["command"] = self.search_info(name)
            self.update()
            if self.famous_count > 194:
                self.famous_count = -1
            if self.famous_count % 28 == 14:
                t_count = 0
                while t_count < 40:
                    if not self.famous:
                        return
                    t_count += 1
                    time.sleep(0.05)
            time.sleep(0.05)

    # ------------------------------------------------------------------------------------------------------------------
    # Global Functions

    @staticmethod
    def show_info(focus):

        def sub_func(event=""):
            info_msg = "----------Cast---------\n(Names are listed in no particular order)\n\n" \
                       "Gan Tan \n" \
                       "He Yujian \n" \
                       "Ma Siyuan \n" \
                       "Zhu Yichen \n"
            tkm.showinfo("made by carefree0910", info_msg)
            focus.focus_set()

        return sub_func

    @staticmethod
    def show_help(focus):

        def sub_func(event=""):
            help_msg = "Shortcuts: \n\n" \
                       "    Ctrl + I      : Focus on searching information entry \n" \
                       "    Ctrl + Q      : Focus on searching departure entry \n" \
                       "    Ctrl + R      : Focus on searching route entry \n" \
                       "    Ctrl + F      : Iconify \n" \
                       "    Ctrl + D      : Help \n" \
                       "    Ctrl + G      : Info \n" \
                       "    Ctrl + P      : Enter Administrator Mode \n" \
                       "    Ctrl + S      : Save Changes in Administrator Mode \n\n" \
                       "    Alt  + Q      : Search Route \n" \
                       "    Alt  + W      : Clear Search History \n\n" \
                       "    Ctrl + Alt + A: Switch priority to 'less cost' \n" \
                       "    Ctrl + Alt + S: Switch priority to 'less time' \n" \
                       "    Ctrl + Alt + D: Switch priority to 'less change' \n\n" \
                       "    Enter         : Take the place of left-button-click in some occasion \n" \
                       "    Esc           : Escape from your current interface. \n\n"
            tkm.showinfo("made by carefree0910", help_msg)
            focus.focus_set()

        return sub_func

    @staticmethod
    def bind_help(obj, focus):
        obj.bind("<Control-d>", TravelGUI.show_help(focus))

    @staticmethod
    def iconify(*top):

        def sub_func(event=""):
            for i in top:
                i.iconify()

        return sub_func

    @staticmethod
    def bind_iconify(obj, *top):
        obj.bind("<Control-f>", TravelGUI.iconify(*top))

    @staticmethod
    def focus_and_select(entry):
        entry.focus_set()
        entry.select_from(0)
        entry.select_to(tk.END)

    def quit_top(self, top):

        def sub_func(event=""):
            top.destroy()
            self.search.focus_set()

        return sub_func

    def close_gui(self):
        result = tkm.askokcancel("made by carefree0910", "Quit?")
        if result:

            # ----------------------------------------------------------------------------------------------------------
            # End Animations
            self.cast = False
            self.famous = False

            time.sleep(0.1)
            while self.cast_thread.is_alive() or self.famous_thread.is_alive():
                time.sleep(0.1)
            tkm.showinfo("made by carefree0910", "See you next time!")
            self.master.destroy()

        else:
            tkm.showinfo("made by carefree0910", "Happy Travelling!")

    # ------------------------------------------------------------------------------------------------------------------
    # Set Vector

    def get_cos(self, target):
        lst = [0] * len(self.total_vector)
        for i in self.vectors[target]:
            lst[i] += 1
        s = 0
        for i in enumerate(self.total_vector):
            s += lst[i[0]] * i[1]
        s /= (TravelGUI.vector_len(lst) * TravelGUI.vector_len(self.total_vector))
        return s

    @staticmethod
    def vector_len(vector):
        s = 0
        for i in vector:
            s += i ** 2
        return math.sqrt(s)

    def save_vector(self):
        with open("Data/Vector.dat", "wb") as file:
            pickle.dump((self.vectors, self.total_vector), file)
        file.close()

    def check_vector(self):
        if self.show_sug or TravelGUI.vector_len(self.total_vector) > 2:
            lst = []
            for i in self.names:
                lst.append((self.get_cos(i), i))
            lst.sort(reverse=True)
            self.show_sug_lst = []
            for i in range(10):
                self.show_sug_lst.append(lst[i][1])
            self.show_sug_box["value"] = self.show_sug_lst
            self.show_sug_box.set(self.show_sug_lst[0])

    # ------------------------------------------------------------------------------------------------------------------
    # Set Departure Point

    def refresh_init(self):
        name = self.init_entry.get().strip()
        self.init_lst = []
        if not name:
            self.init_lst = list(self.names.keys())
        else:
            cursor = -1
            equal = False
            for i in self.names:
                judge = True
                for j in name:
                    if j not in i:
                        judge = False
                        break
                if judge:
                    if name == i:
                        equal = True
                        cursor = len(self.init_lst)
                    elif name in i and not equal:
                        cursor = len(self.init_lst)
                    self.init_lst.append(i)
            if cursor > 0:
                t = self.init_lst.pop(cursor)
                self.init_lst.insert(0, t)
        self.init_choice_box["value"] = self.init_lst
        if not self.init_lst:
            self.init_choice_box.set("None")
        else:
            self.init_choice_box.set(self.init_lst[0])

    def set_init_spot(self):
        spot = self.init_choice_box.get().strip()
        if spot == "None":
            tkm.showerror("made by carefree0910", "Please provide us with the departure spot!")
        else:
            tkm.showinfo("made by carefree0910", "You've set {} as your departure spot!".format(spot))
            self.departure = spot

    # ------------------------------------------------------------------------------------------------------------------
    # Search Route

    def add_search_spot(self):
        now = self.search_str.get()
        if now == "Target not selected yet":
            now = ""
        lst = now.split(", ")
        new = self.search.get().strip()
        if new and new not in lst:
            self.search_str.set("{} {}".format("{},".format(now) if now else "", new).strip())
        TravelGUI.focus_and_select(self.search)

    def del_search_spot(self):
        if self.search_str.get() != "Target not selected yet":
            lst = self.search_str.get().strip().split(", ")
            if not lst:
                self.search.focus_set()
            else:
                top = tk.Toplevel(self.master)
                top.title("Delete Spots")
                top.bind("<Escape>", lambda event: top.destroy())
                TravelGUI.bind_iconify(top, top)

                s_c = ttk.Combobox(top, value=lst, width=36, state="readonly")
                s_c.set(lst[0])
                s_b = tk.Button(top, text="Delete", width=34, bd=0, bg="LightCoral",
                                command=TravelGUI.confirm_del(self.master, top, s_c, lst, self.search_str))
                s_c.grid()
                s_b.grid()
                TravelGUI.focus_and_select(self.search)

    def search_route(self):
        self.add_search_spot()
        target = self.search_str.get().strip()
        if not target or target == "Target not selected yet":
            tkm.showerror("made by carefree0910", "Please provide us with a suitable target spot!")
        elif self.init_choice_box.get() == "None":
            tkm.showerror("made by carefree0910", "Please provide us with a suitable departure spot!")
        else:
            target = list(set(target.split(", ")))
            all_lst = []
            for i in target:
                t_lst = []
                cursor = -1
                equal = False
                for j in self.names:
                    judge = True
                    for k in i:
                        if k not in j:
                            judge = False
                            break
                    if judge:
                        if i == j:
                            equal = True
                            cursor = len(t_lst)
                        elif i in j and not equal:
                            cursor = len(t_lst)
                        t_lst.append(j)
                if not t_lst:
                    tkm.showerror("made by carefree0910", "We can't find '{}' in our database...".format(i))
                    TravelGUI.focus_and_select(self.search)
                else:
                    if cursor >= 0:
                        t = t_lst.pop(cursor)
                        t_lst.insert(0, t)
                    all_lst.append(t_lst)
            if not all_lst:
                tkm.showerror("made by carefree0910", "None of '{}' is in our database...".format(target))
            else:
                self.start_route(all_lst)

    def start_route(self, all_lst):
        length = len(all_lst)
        for i in enumerate(all_lst):
            top = tk.Toplevel(self.master)
            top.bind("<Escape>", self.quit_top(top))
            TravelGUI.bind_iconify(top, top)
            t_frame = tk.LabelFrame(top, text="Choose your target!")

            last = False if i[0] != length - 1 else True
            t_box = ttk.Combobox(t_frame, width=42, value=i[1], state="readonly")
            t_box.set(i[1][0])
            TravelGUI.bind_help(top, t_box)
            txt = tk.StringVar()
            if last:
                txt.set("Go!")
            else:
                txt.set("Next!")
            t_button = tk.Button(t_frame, textvariable=txt, width=12, bd=0, bg="FloralWhite",
                                 command=self.t_confirm(top, t_box, last))
            t_button.bind("<Return>", self.t_confirm(top, t_box, last))

            t_frame.grid()
            t_box.grid(row=0, column=0)
            t_button.grid(row=0, column=1)
            t_button.focus_set()

    def t_confirm(self, top, t_box, last):

        def sub_func(event=""):
            target = t_box.get()
            top.destroy()
            self.targets.add(target)
            if not last:
                pass
            else:
                command = self.prior.get().split()[1]
                self.complete_route(command)

        return sub_func

    def complete_route(self, command):
        length = len(self.targets)
        if self.departure in self.targets and length == 1:
            tkm.showinfo("made by carefree0910", "You're exactly at spot '{}' now...".format(self.departure))
        else:
            self.targets.discard(self.departure)
            target = list(self.targets)

            # Modify Vectors
            for i in target:
                for j in self.vectors[i].items():
                    self.total_vector[j[0]] += j[1]
            self.save_vector()
            self.check_vector()

            target.insert(0, self.departure)
            self.targets.clear()
            result = self.get_route(target, command)
            prior = "less {}".format(command)
            tkm.showinfo("Search Result",
                         "Your command: {}\n\n-> Result:\n{}".format(prior, result))

    def get_route(self, target, command):

        class SubThread:
            data = {}

            @staticmethod
            def execute(dic):
                nonlocal target, command
                result = Gen.get_ways(target, command, dic)
                SubThread.data["new"] = result

        class MainThread:

            @staticmethod
            def call_func(names, top):
                nonlocal target, command
                self.master.protocol("WM_DELETE_WINDOW", lambda: None)
                obj = SubThread()
                thread = threading.Thread(target=obj.execute, args=(names,))
                thread.start()
                btn = tk.Label(top, bd=0, bg="Silver", text="Searching...", width=12)
                btn.grid(row=2, columnspan=20)
                count = 16
                while thread.is_alive():
                    count += 1
                    if count < 35:
                        btn["width"] += 2
                    if (100 > count > 50 and count % 4 == 0) or (200 > count > 150 and count % 6 == 0) or \
                            (count > 250 and count % 16 == 0):
                        btn["width"] += 2
                    time.sleep(0.1)
                    top.update()
                btn.destroy()
                self.master.protocol("WM_DELETE_WINDOW", self.close_gui)
                resp = SubThread.data["new"]
                del SubThread.data["new"]
                return resp

        return MainThread.call_func(self.names, self.all_frame)

    # ------------------------------------------------------------------------------------------------------------------
    # Search Information

    def search_info(self, famous=""):

        def sub_func(event=""):
            if not famous:
                name, t_type, loc = self.search_name.get().strip(), self.search_type.get(), self.search_loc.get()
            else:
                name, t_type, loc = famous, "Type", "Location"
            info_lst = []

            if name == "Name" or not name:
                name = "All"
                for i in self.names:
                    nxt = self.names[i]
                    if (t_type != "Type" and t_type not in nxt[1]) or (loc != "Location" and loc not in nxt[0]):
                        continue
                    info_lst.append((i, self.names[i]))
                if not info_lst:
                    tkm.showerror("made by carefree0910", "What you're searching for is not in the database...")
                    TravelGUI.focus_and_select(self.search_name)
                    return

            else:
                cursor = -1
                if not famous:
                    equal = False
                    for i in self.names:
                        judge = True
                        for j in name:
                            if j not in i:
                                judge = False
                                break
                        if judge:
                            nxt = self.names[i]
                            if (t_type != "Type" and t_type not in nxt[1]) or (loc != "Location" and loc not in nxt[0]):
                                continue
                            if name == i:
                                equal = True
                                cursor = len(info_lst)
                            elif name in i and not equal:
                                cursor = len(info_lst)
                            info_lst.append((i, self.names[i]))
                else:
                    info_lst = [(name, self.names[name])]

                if not info_lst:
                    tkm.showerror("made by carefree0910", "What you're searching for is not in the database...")
                    TravelGUI.focus_and_select(self.search_name)
                    return
                if cursor > 0:
                    t = info_lst.pop(cursor)
                    info_lst.insert(0, t)

            info_str = "{}{}{}{}{}" \
                .format(name, " with " if t_type != "Type" else "", t_type if t_type != "Type" else "",
                        " in " if loc != "Location" else "", loc if loc != "Location" else "")
            self.start_info(info_str, info_lst, famous)

        return sub_func

    def start_info(self, info_str, info_lst, famous=""):
        top = tk.Toplevel(self.master)
        top.title("Search Result")
        top.bind("<Escape>", self.quit_top(top))
        TravelGUI.bind_iconify(top, top)

        save1 = " and modify it" if self.admin else ""
        save2 = "details"
        if self.admin:
            save2 = "details and change them"

        info_frame = tk.LabelFrame(top)
        info_scroll = tk.Scrollbar(info_frame, orient=tk.VERTICAL)
        info_scroll.pack(side="right", fill="y")
        info_box = tk.Listbox(info_frame, width=42, height=20, yscrollcommand=info_scroll.set)
        info_box.insert(0, "Left-Click to get brief info{};".format(save1))
        info_box.insert(1, "Double-Click to get {}!".format(save2))
        info_box.insert(2, "")
        for i in enumerate(info_lst):
            info_box.insert(i[0] + 3, i[1][0])
        info_box.pack(side="left", fill="both")
        info_scroll.config(command=info_box.yview)
        info_frame.grid(row=1, column=0, rowspan=4, columnspan=2)
        TravelGUI.bind_help(top, info_box)

        if self.admin:
            info_str += "; press buttons on the right to add/delete spots"
        name_frame = tk.LabelFrame(top)
        name_label = tk.Label(name_frame, text="Searching for: {}".format(info_str), width=70, bd=0)
        name_add = tk.Button(name_frame, image=self.add_pic, command=self.add_spot(info_lst, info_box), bd=0)
        name_del = tk.Button(name_frame, image=self.del_pic, command=self.del_spot(info_lst, info_box), bd=0)
        name_frame.grid(row=0, columnspan=4)
        name_label.grid(row=0, column=0)
        if self.admin:
            name_label["width"] = 66
            name_add.grid(row=0, column=1)
            name_del.grid(row=0, column=2)

        label_frame = tk.LabelFrame(top, text="Labels")
        label_str = tk.StringVar()
        label_str.set("Not selected yet")
        if not self.admin:
            label_content = tk.Label(label_frame, textvariable=label_str, bd=0)
        else:
            label_content = tk.Entry(label_frame, textvariable=label_str, bd=0, width=32)
            label_content.bind("<Button-1>", self.switch_admin_cursor(info_box))

        label_save = tk.Button(label_frame, image=self.save_pic,
                               command=self.save_label(info_box, label_content), bd=0)

        cord_frame = tk.LabelFrame(top, text="Coordinate")
        x_str, y_str = tk.StringVar(), tk.StringVar()
        x_str.set("N: ")
        y_str.set("E: ")
        x_entry = tk.Entry(cord_frame, textvariable=x_str, width=28, bd=0, justify="center", bg="Ivory")
        y_entry = tk.Entry(cord_frame, textvariable=y_str, width=28, bd=0, justify="center", bg="Ivory")
        x_save = tk.Button(cord_frame, image=self.save_pic, command=self.save_pos(info_box, x_entry, 0), bd=0)
        y_save = tk.Button(cord_frame, image=self.save_pic, command=self.save_pos(info_box, y_entry, 1), bd=0)
        x_entry.grid(row=0, column=0)
        y_entry.grid(row=1, column=0)
        if self.admin:
            x_entry["width"] = y_entry["width"] = 26
            x_save.grid(row=0, column=1)
            y_save.grid(row=1, column=1)
        if not self.special_admin:
            x_entry["state"] = y_entry["state"] = x_save["state"] = y_save["state"] = "disabled"
        adr_frame = tk.LabelFrame(top, text="Address")
        adr_scroll = tk.Scrollbar(adr_frame, orient=tk.VERTICAL)
        adr_box = tk.Text(adr_frame, width=26, height=4, bd=0, bg="Linen", yscrollcommand=adr_scroll.set)
        adr_box.insert(0.0, "Not selected yet")
        adr_box.pack(side="left", fill="both")
        adr_scroll.config(command=adr_box.yview)
        adr_save = tk.Button(adr_frame, image=self.save_pic, command=self.save_adr(info_box, adr_box), bd=0)
        if self.admin:
            adr_box["width"] = 28
            adr_box.bind("<Button-1>", self.switch_admin_cursor(info_box))
            adr_scroll.pack(side="top", fill="y")
            adr_save.pack(side="bottom")
        else:
            adr_scroll.pack(side="right", fill="y")

        tran_frame = tk.LabelFrame(top, text="Transportation")
        tran_scroll = tk.Scrollbar(tran_frame, orient=tk.VERTICAL)
        tran_box = tk.Text(tran_frame, width=26, height=10, bd=0, bg="GhostWhite",
                           yscrollcommand=tran_scroll.set)
        tran_box.insert(0.0, "Not selected yet")
        tran_box.pack(side="left", fill="both")
        tran_scroll.config(command=tran_box.yview)
        tran_save = tk.Button(tran_frame, image=self.save_pic, state="disabled",
                              command=self.save_tran(info_box, tran_box), bd=0)
        tran_m_f = tk.LabelFrame(top, text="Modify your transportation here")
        tran_c_l = tk.Label(tran_m_f, text="   You may modify the buses' information, but not the subways'...   ")
        tran_c_b = ttk.Combobox(tran_m_f, width=6, state="readonly")
        tran_c_a = tk.Button(tran_m_f, image=self.add_pic, bd=0, command=self.add_bus(top, tran_c_b, info_box))
        tran_c_d = tk.Button(tran_m_f, image=self.del_pic, bd=0, command=self.del_bus(tran_c_b, info_box))
        tran_c_l.grid(row=0, column=0)
        tran_c_b.grid(row=0, column=1)
        tran_c_a.grid(row=0, column=2)
        tran_c_d.grid(row=0, column=3)
        if self.admin:
            tran_box["width"] = 28
            tran_box["height"] = 4
            tran_box.bind("<Button-1>", self.switch_admin_cursor(info_box))
            tran_scroll.pack(side="top")
            tran_save.pack(side="bottom")
            tran_m_f.grid(row=5, columnspan=3)
        else:
            tran_scroll.pack(side="right", fill="y")

        label_frame.grid(row=1, column=2)
        label_content.grid(row=0, column=0)
        if self.admin:
            label_save.grid(row=0, column=1)
        cord_frame.grid(row=2, column=2)
        if self.admin:
            x_entry["width"] = y_entry["width"] = 30
        adr_frame.grid(row=3, column=2)

        tran_frame.grid(row=4, column=2)

        info_box.bind("<ButtonRelease-1>", self.first_info(info_box, info_lst, label_str,
                                                           adr_box, tran_box, x_str, y_str, tran_c_b))
        info_box.bind("<KeyRelease>", self.first_info(info_box, info_lst, label_str,
                                                      adr_box, tran_box, x_str, y_str, tran_c_b))
        info_box.bind("<Double-Button-1>", self.second_info(info_box, info_lst, top))
        info_box.bind("<Return>", self.second_info(info_box, info_lst, top))
        info_box.focus_set()
        if famous:
            info_box.selection_set(3)
            self.first_info(info_box, info_lst, label_str, adr_box, tran_box, x_str, y_str, tran_c_b)()

    def first_info(self, info_box, info_lst, label_str, adr_box, tran_box, x_str, y_str, tran_c_b):

        def sub_func(event=""):
            cursor = info_box.curselection()
            if not cursor or cursor[0] <= 2:
                self.admin_cursor = 0
                label_str.set("Not selected yet")
                adr_box.delete(0.0, tk.END)
                adr_box.insert(0.0, "Not selected yet")
                tran_box.delete(0.0, tk.END)
                tran_box.insert(0.0, "Not selected yet")
                x_str.set("N: ")
                y_str.set("E: ")
            else:
                self.admin_cursor = cursor[0]
                item = info_lst[cursor[0] - 3][1]
                label_str.set(" ".join(set(item[1])).strip())
                adr_box.delete(0.0, tk.END)
                adr_box.insert(0.0, "    {}".format(item[0]))
                tran_box.delete(0.0, tk.END)
                tran_box.insert(0.0, "    {}".format(item[3]))
                name = info_lst[cursor[0] - 3][0]
                lst = self.names[name][5].split()
                tran_c_b["value"] = lst
                if not lst:
                    tran_c_b.set("None")
                else:
                    tran_c_b.set(lst[0])

                x_str.set("N: {}".format(item[4][0]))
                y_str.set("E: {}".format(item[4][1]))

        return sub_func

    def second_info(self, info_box, info_lst, top):

        def sub_func(event=""):
            cursor = info_box.curselection()
            if not cursor or cursor[0] <= 2:
                pass
            else:
                name, details = info_lst[cursor[0] - 3][0], info_lst[cursor[0] - 3][1][2]
                sub_top = tk.Toplevel(top)
                TravelGUI.bind_iconify(sub_top, sub_top)
                sub_top.bind("<Escape>", lambda ev: sub_top.destroy())

                detail_scroll = tk.Scrollbar(sub_top, orient=tk.VERTICAL)
                detail_box = tk.Text(sub_top, height=10, yscrollcommand=detail_scroll.set, bg="Ivory")
                detail_box.insert(0.0, "    {}".format(details))
                detail_scroll.config(command=detail_box.yview)
                TravelGUI.bind_help(sub_top, detail_box)

                if not self.admin:

                    # Modify Vectors
                    for i in self.vectors[name].items():
                        self.total_vector[i[0]] += 0.8 * i[1]
                    self.save_vector()
                    self.check_vector()

                    sub_top.title("Details of {}".format(name))
                    detail_scroll.pack(side="right", fill="y")
                    detail_box.pack()
                    switch_frame = tk.LabelFrame(sub_top)
                    switch_label = tk.Label(switch_frame, bg="Coral", fg="white", width=60,
                                            text="Want to travel here? Click the button to go searching for the route!")
                    switch_prior = ttk.Combobox(switch_frame, width=19, state="readonly",
                                                values=["less cost", "less time", "less change"])
                    switch_prior.set("less cost")
                    switch_button = tk.Button(switch_frame, width=18, text="Go!",
                                              command=self.switch_to_route(name, top, switch_prior),
                                              bd=0, bg="MintCream")
                    switch_frame.pack(side="bottom", fill="x")
                    switch_label.pack(side="left", fill="y")
                    switch_button.pack(fill="x")
                    switch_prior.pack(side="right", fill="y")
                    detail_box.focus_set()
                else:
                    sub_top.title("Details of {}".format(name))
                    detail_scroll.pack(side="right", fill="y")
                    detail_box.pack()
                    admin_frame = tk.LabelFrame(sub_top)
                    admin_label = tk.Label(admin_frame, bg="Coral", fg="white", width=58, height=2, bd=2,
                                           text="Click the button on the right to confirm your modification!")
                    admin_button = tk.Button(admin_frame, width=20, text="Modify!",
                                             command=self.save_detail(name, sub_top, detail_box, info_box),
                                             bd=0, bg="MintCream")
                    detail_box["height"] = 14
                    admin_frame.pack(side="bottom", fill="x")
                    admin_label.pack(side="left", fill="y")
                    admin_button.pack(side="right", fill="y")
                    detail_box.focus_set()

        return sub_func

    def switch_to_route(self, n, top=None, prior=None):

        def sub_func():
            name = n.get() if top is None and prior is None else n

            # Modify Vectors
            for i in self.vectors[name].items():
                self.total_vector[i[0]] += 0.2 * i[1]
            self.save_vector()
            self.check_vector()

            self.search.focus_set()
            self.search.delete(0, tk.END)
            self.search.insert(0, name)
            self.add_search_spot()
            if top is not None and prior is not None:
                self.prior.set(prior.get())
                top.destroy()

        return sub_func

    # ------------------------------------------------------------------------------------------------------------------
    # Administrator

    def create_admin_top(self):
        if self.admin:
            self.quit_admin()
        else:
            top = tk.Toplevel(self.master)
            top.title("")
            top.bind("<Escape>", lambda event: top.destroy())
            TravelGUI.bind_iconify(top, top)
            password_label = tk.Label(top, text="Password: ", width=8)
            password = tk.Entry(top, width=16, show="*", bd=0)
            password.bind("<Return>", self.check_password(top, password))
            password_label.grid(row=0, column=0)
            password.grid(row=0, column=1)
            tkm.showinfo("made by carefree0910", "Please enter the password first.")
            password.focus_set()

    def check_password(self, top, password):

        def sub_func(event=""):
            word = password.get()
            if word in (self.password, self.special_password):
                top.destroy()
                if word == self.special_password:
                    self.special_admin = True
                self.admin_save.grid()
                self.admin_to_spot()
            else:
                tkm.showerror("made by carefree0910", "Wrong password!")
                self.search_name.focus_set()
                top.destroy()

        return sub_func

    def admin_to_spot(self):
        self.admin = True
        self.admin_str.set("Exit")
        self.master.title("Administrator Mode")
        self.search_info_frame["text"] = "Search for spot to modify it"
        TravelGUI.focus_and_select(self.search_name)

    def switch_admin_cursor(self, info_box):

        def sub_func(event=""):
            cursor = info_box.curselection()
            if cursor:
                self.admin_cursor = cursor[0]

        return sub_func

    def save_label(self, info_box, label_content):

        def sub_func():
            if self.admin_cursor < 3:
                return
            name = info_box.get(self.admin_cursor)
            labels = label_content.get().split()
            self.names[name][1] = labels
            for i in labels:
                if i not in self.label_set:
                    self.label_set.add(i)
                    del self.labels[0]
                    self.labels.append(i)
                    self.labels.sort()
                    self.labels.insert(0, "Type")

                    # Modify Vectors
                    self.total_vector.append(0)
                    self.vectors[name] = {}
                    self.vectors = Gen.get_vec(self.vectors, {name: self.names[name]}, self.labels, self.label_set)
                    self.save_vector()

            self.search_type["value"] = self.labels
            tkm.showinfo("made by carefree0910", "You've changed {}'s label to {}!".format(name, labels))
            info_box.focus_set()
            info_box.activate(self.admin_cursor)

        return sub_func

    def save_pos(self, info_box, entry, pos):

        def sub_func():
            if self.admin_cursor < 3:
                return
            name = info_box.get(self.admin_cursor)
            try:
                grid = str(float(entry.get()[2:].strip()))
                self.names[name][4][pos] = grid
                tkm.showinfo("made by carefree0910", "You've changed {}'s '{}' to {}!"
                             .format(name, "N" if pos == 0 else "E", grid))
            except ValueError as err1:
                tkm.showerror("made by carefree0910", "Error occurred when trying to change '{}':\n-> {}"
                              .format("N" if pos == 0 else "E", err1))
            finally:
                info_box.focus_set()
                info_box.activate(self.admin_cursor)

        return sub_func

    def save_adr(self, info_box, adr_box):

        def sub_func():
            if self.admin_cursor < 3:
                return
            name = info_box.get(self.admin_cursor)
            adr = adr_box.get(0.0, tk.END).replace("\n", "    ").strip()
            self.names[name][0] = adr
            tkm.showinfo("made by carefree0910", "You've changed {}'s address to {}!".format(name, adr))
            info_box.focus_set()
            info_box.activate(self.admin_cursor)

        return sub_func

    def save_tran(self, info_box, tran_box):

        def sub_func():
            if self.admin_cursor < 3:
                return
            name = info_box.get(self.admin_cursor)
            tran = tran_box.get(0.0, tk.END).replace("\n", "    ").strip()
            self.names[name][3] = tran
            tkm.showinfo("made by carefree0910", "You've changed {}'s transportation to {}!".format(name, tran))
            info_box.focus_set()
            info_box.activate(self.admin_cursor)

        return sub_func

    def save_detail(self, name, sub_top, detail_box, info_box):

        def sub_func():
            if self.admin_cursor < 3:
                return
            detail = detail_box.get(0.0, tk.END).replace("\n", "    ").strip()
            self.names[name][2] = detail
            tkm.showinfo("made by carefree0910", "You've changed {}'s detailed information!".format(name))
            sub_top.destroy()
            info_box.focus_set()

        return sub_func

    @staticmethod
    def add_label(l_c, l_var):

        def sub_func():
            l_var.set(("{} {}".format(l_var.get(), l_c.get().strip())).strip())

        return sub_func

    @staticmethod
    def del_label(top, l_var):

        def sub_func():
            lst = l_var.get().strip().split()
            if not lst:
                top.focus_set()
            else:
                sub_top = tk.Toplevel(top)
                sub_top.title("Delete Label")
                sub_top.bind("<Escape>", lambda event: sub_top.destroy())
                TravelGUI.bind_iconify(sub_top, sub_top)

                l_c = ttk.Combobox(sub_top, value=lst, width=16, state="readonly")
                l_c.set(lst[0])
                l_b = tk.Button(sub_top, text="Delete", bd=0, bg="LightCoral",
                                command=TravelGUI.confirm_del(top, sub_top, l_c, lst, l_var), width=17)
                l_c.grid()
                l_b.grid()

        return sub_func

    @staticmethod
    def confirm_del(top, sub_top, l_c, lst, l_var):

        def sub_func():
            del lst[lst.index(l_c.get().strip())]
            if len(lst) == 0:
                l_var.set("Target not selected yet")
            elif len(lst) == 1:
                l_var.set(lst[0])
            else:
                l_var.set(", ".join(lst))
            sub_top.destroy()
            top.focus_set()

        return sub_func

    @staticmethod
    def switch_tran(frame, box, entry):

        def sub_func(event=""):
            target = box.get()
            if target == "walk":
                entry.delete(0, tk.END)
                entry["state"] = "disabled"
                frame["text"] = "No need to fill this"
            else:
                entry.delete(0, tk.END)
                entry["state"] = "normal"
                frame["text"] = "Which one?"
                show = "1号线" if target == "subway" else "1"
                entry.insert(0, show)
                TravelGUI.focus_and_select(entry)

        return sub_func

    def add_spot(self, info_lst, info_box):

        def sub_func():
            top = tk.Toplevel(self.master)
            top.title("Add Spot")
            top.bind("<Escape>", lambda event: top.destroy())
            TravelGUI.bind_iconify(top, top)

            n_f = tk.LabelFrame(top, text="Name")
            l_f = tk.LabelFrame(top, text="Labels")
            p_f = tk.LabelFrame(top, text="Coordinate")
            a_f = tk.LabelFrame(top, text="Address")
            t_f = tk.LabelFrame(top, text="Buses (1 ~ 999)")
            i_f = tk.LabelFrame(top, text="Information")

            n_e = tk.Entry(n_f, width=27)
            l_var = tk.StringVar()
            l_l = tk.Label(l_f, textvariable=l_var, width=24)
            l_e = tk.Entry(l_f, width=18)
            l_a = tk.Button(l_f, image=self.add_pic, command=TravelGUI.add_label(l_e, l_var), bd=0)
            l_d = tk.Button(l_f, image=self.del_pic, command=TravelGUI.del_label(top, l_var), bd=0)
            p_l_n = tk.Label(p_f, text="N: (39.4 ~ 41.1)")
            p_l_e = tk.Label(p_f, text="E: (115.4 ~ 117.5)")
            p_e_n = tk.Entry(p_f, width=12, justify="center")
            p_e_e = tk.Entry(p_f, width=12, justify="center")
            a_b = tk.Text(a_f, width=24, height=6)
            t_var = tk.StringVar()
            t_l = tk.Label(t_f, textvariable=t_var, width=24)
            t_e = tk.Entry(t_f, width=23, justify="center")
            t_a = tk.Button(t_f, image=self.add_pic, command=TravelGUI.add_label(t_e, t_var), bd=0)
            t_d = tk.Button(t_f, image=self.del_pic, command=TravelGUI.del_label(top, t_var), bd=0)
            i_b = tk.Text(i_f, width=40, height=12)

            n_f.grid(row=0, column=0)
            l_f.grid(row=0, column=1, rowspan=2)
            p_f.grid(row=1, column=0)
            a_f.grid(row=2, column=0)
            t_f.grid(row=3, column=0, rowspan=2)
            i_f.grid(row=2, column=1, rowspan=2)

            n_e.grid()
            l_l.grid()
            l_e.grid(row=1, column=0)
            l_a.grid(row=1, column=1)
            l_d.grid(row=1, column=2)
            p_l_n.grid(row=0, column=0)
            p_e_n.grid(row=0, column=1)
            p_l_e.grid(row=1, column=0)
            p_e_e.grid(row=1, column=1)
            a_b.grid()
            t_l.grid(row=0, column=0, columnspan=3)
            t_e.grid(row=1, column=0)
            t_a.grid(row=1, column=1)
            t_d.grid(row=1, column=2)
            i_b.grid()
            n_e.focus_set()

            confirm_btn = tk.Button(top, text="Confirm",
                                    command=self.finish_add_spot(top, info_lst, info_box, n_e, l_var, p_e_n, p_e_e,
                                                                 a_b, t_var, i_b), width=40, bg="PaleGreen")
            confirm_btn.grid(row=4, column=1)

        return sub_func

    def finish_add_spot(self, top, info_lst, info_box, n_e, l_var, p_e_n, p_e_e, a_b, t_var, i_b):

        def sub_func():
            name, label, pos, adr, tran, info = n_e.get().strip(), l_var.get().split(), \
                                                [p_e_n.get().strip(), p_e_e.get().strip()], \
                                                a_b.get(0.0, tk.END).strip(), t_var.get().strip(), \
                                                i_b.get(0.0, tk.END).strip()
            if not name:
                name = "None"
            if not label:
                label = ["None"]
            if not pos:
                pos = ["None", "None"]
            if not adr:
                adr = "None"
            if not tran:
                tran = ""
            if not info:
                info = "None"
            try:
                pos = [float(pos[0]), float(pos[1])]
                judge = tkm.askokcancel("made by carefree0910", "Confirm addition?")
                if not judge:
                    top.focus_set()
                elif pos[0] < 39.4 or pos[0] > 41.1 or pos[1] < 115.4 or pos[1] > 117.5:
                    tkm.showerror("made by carefree0910", "Please add a spot in Beijing...")
                    top.focus_set()
                else:
                    pos = [str(pos[0]), str(pos[1])]
                    j = True
                    for i in info_lst:
                        if i[0] == name:
                            tkm.showwarning("made by carefree0910", "Spot '{}' is concluded in our database already!"
                                            .format(name))
                            j = False
                            top.focus_set()
                            break
                    if j:
                        name = name.replace("\n", "    ").strip()
                        adr = adr.replace("\n", "    ").strip()
                        info = info.replace("\n", "    ").strip()
                        vehicle = tran
                        tran = "公交：" + "、".join(tran.split())

                        self.names[name] = [adr, label, info, tran, pos, vehicle]
                        for k in label:
                            if k not in self.label_set:
                                self.label_set.add(k)
                                del self.labels[0]
                                self.labels.append(k)
                                self.labels.sort()
                                self.labels.insert(0, "Type")

                                # Modify Vectors
                                self.vectors[name] = {}
                                self.total_vector.append(0)
                                self.vectors = Gen.get_vec(self.vectors, {name: self.names[name]},
                                                           self.labels, self.label_set)
                        self.search_type["value"] = self.labels
                        self.save_vector()
                        self.check_vector()

                        info_box.insert(tk.END, name)
                        info_lst.append((name, [adr, label, info, tran, pos, vehicle]))
                        info_box.focus_set()
                        top.destroy()
            except ValueError as err1:
                tkm.showerror("made by carefree0910", err1)
                top.focus_set()

        return sub_func

    def del_spot(self, info_lst, info_box):

        def sub_func(event=""):
            if self.admin_cursor < 3:
                tkm.showerror("made by carefree0910", "Please select a spot to delete!")
                info_box.focus_set()
            else:
                name = info_box.get(self.admin_cursor)
                if name in self.famous_name_lst:
                    tkm.showwarning("made by carefree0910",
                                    "'{}' is one of the famous spots you're advertising, please don't delete it..."
                                    .format(name))
                else:
                    judge = tkm.askokcancel("made by carefree0910", "Delete spot '{}'?".format(name))
                    if not judge:
                        info_box.focus_set()
                    else:
                        info_box.delete(self.admin_cursor)
                        info_box.select_set(0)
                        info_box.activate(0)
                        del self.names[name]
                        del info_lst[self.admin_cursor - 3]
                        self.admin_cursor = 0
                        info_box.focus_set()

        return sub_func

    def add_bus(self, top, box, info_box):

        def sub_func():
            if self.admin_cursor < 3:
                return
            name = info_box.get(self.admin_cursor)
            sub_top = tk.Toplevel(top)
            label = tk.Label(sub_top, text="Please input the bus_number here(1 ~ 999)")
            entry = tk.Entry(sub_top, bd=1, width=35, justify="center")
            button = tk.Button(sub_top, image=self.confirm_pic, bd=0,
                               command=self.confirm_bus(name, entry, box, sub_top))
            entry.bind("<Return>", self.confirm_bus(name, entry, box, sub_top))
            label.grid(row=0, column=0, columnspan=2)
            entry.grid(row=1, column=0)
            button.grid(row=1, column=1)
            entry.focus_set()

        return sub_func

    def confirm_bus(self, name, entry, box, sub_top):

        def sub_func(event=""):
            try:
                num = int(entry.get().strip())
                if not 1 <= num <= 999:
                    tkm.showerror("made by carefree0910", "bus_number should be in [1, 999]...")
                    TravelGUI.focus_and_select(entry)
                else:
                    lst = list(box["value"])
                    lst.append(str(num))
                    box["value"] = lst
                    box.set(lst[0])
                    self.names[name][5] = " ".join(lst)
                    sub_top.destroy()
            except ValueError as err:
                tkm.showerror("made by carefree0910", err)
                TravelGUI.focus_and_select(entry)

        return sub_func

    def del_bus(self, box, ib):

        def sub_func():
            if self.admin_cursor < 3:
                return
            name = ib.get(self.admin_cursor)
            lst = list(box["value"])
            now = box.get()
            if now == "None":
                tkm.showerror("made by carefree0910", "Nothing to delete now...")
                box.focus_set()
            else:
                try:
                    float(now)
                    ans = tkm.askquestion("made by carefree0910", "Confirm deletion of {} bus?".format(now))
                    if ans == "no":
                        box.focus_set()
                    else:
                        del lst[lst.index(now)]
                        self.names[name][5] = " ".join(lst)
                        box["value"] = lst
                        if not lst:
                            box.set("None")
                        else:
                            box.set(lst[0])
                        box.focus_set()
                except ValueError:
                    tkm.showerror("made by carefree0910", "This is the subway line, please don't delete it...")
                    box.focus_set()

        return sub_func

    def quit_admin(self):
        judge = tkm.askokcancel("made by carefree0910", "Quit Administrator Mode?")
        if not judge:
            pass
        else:
            self.special_admin = False
            self.admin_save.grid_remove()
            self.admin_str.set("Enter")
            self.master.title("Travel Helper")
            self.search_info_frame["text"] = "Search for information"

            self.save()
            self.admin = False

    def save(self):

        if self.admin:

            def sub_save():
                time.sleep(0.5)
                with open("Data/Info.dat", "wb") as file:
                    pickle.dump((self.names, self.labels, self.label_set, self.locs, self.loc_set), file)
                file.close()

            class MainThread:

                @staticmethod
                def call_func(top):
                    self.master.protocol("WM_DELETE_WINDOW", lambda: None)
                    thread = threading.Thread(target=sub_save)
                    thread.start()
                    btn = tk.Label(top, bd=0, bg="Silver", text="Saving...", width=16)
                    btn.grid(row=2, columnspan=20)
                    while thread.is_alive():
                        btn["width"] += 4
                        top.update()
                        time.sleep(0.1)
                    btn.destroy()
                    self.master.protocol("WM_DELETE_WINDOW", self.close_gui)

            MainThread.call_func(self.all_frame)

    # ------------------------------------------------------------------------------------------------------------------
    # Utilities

    def test(self):
        pass


if __name__ == '__main__':
    try:
        TravelGUI().mainloop()
    except Exception as err:
        tkm.showwarning("made by carefree0910", "Oops, some exception occurred...\n-> {}".format(err))
        TravelGUI().mainloop()
