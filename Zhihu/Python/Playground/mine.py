from random import randint
from math import sqrt
import tkinter as tk
import tkinter.messagebox as tkm


class Game(tk.Frame):

    # ==================================================================================================================
    # Cores
    # ------------------------------------------------------------------------------------------------------------------
    # First Core: Used to dig_out the game correctly
    # ------------------------------------------------------------------------------------------------------------------
    # Core Function 1_1

    @staticmethod
    def check_mine(x, y, list1, list2, row, column):

        """Check whether mines there and Game.count"""

        mine_set = set()
        t_bool = False
        c_mine = 0
        for i in range(x - 1, x + 2):
            if i < 0:
                continue
            elif i >= row:
                continue
            else:
                for j in range(y - 1, y + 2):
                    if j < 0:
                        continue
                    elif j >= column:
                        continue
                    else:
                        mine_set.add((i, j))

        for i in mine_set:
            if list2[x][y] == "P":
                t_bool = False
            else:
                if list1[i[0]][i[1]] == 1:
                    t_bool = True
                    c_mine += 1
        return t_bool, c_mine

    # ------------------------------------------------------------------------------------------------------------------
    # Core Function 1_2    

    @staticmethod
    def dig_out(x, y, list1, list2, row, column, all_set):

        """ dig_out what I want """

        temp_set = set()

        (t_bool, c_mine) = Game.check_mine(x, y, list1, list2, row, column)
        if t_bool:
            list2[x][y] = str(c_mine)
            all_set.add((x, y))
            Game.helper(list1, list2, x, y, row, column, all_set)
            return
        else:
            list2[x][y] = "0"
            all_set.add((x, y))
            for i in range(x - 1, x + 2):
                if i < 0:
                    continue
                elif i >= row:
                    continue
                else:
                    for j in range(y - 1, y + 2):
                        if j < 0:
                            continue
                        elif j >= column:
                            continue
                        elif (i, j) == (x, y):
                            continue
                        elif list1[i][j] == 1:
                            continue
                        else:
                            (t_bool, c_mine) = Game.check_mine(i, j, list1, list2, row, column)
                            if t_bool:
                                list2[i][j] = str(c_mine)
                            else:
                                temp_set.add((i, j))
            for temp in temp_set - all_set:
                temp0 = temp[0]
                temp1 = temp[1]
                (t_bool, c_mine) = Game.check_mine(temp0, temp1, list1, list2, row, column)
                if t_bool:
                    list2[temp0][temp1] = str(c_mine)
                    continue
                else:
                    Game.dig_out(temp0, temp1, list1, list2, row, column, all_set)
            return

    # ------------------------------------------------------------------------------------------------------------------
    # Second Core: Make the game more playable
    # ------------------------------------------------------------------------------------------------------------------
    # Core Function 2_1_1

    @staticmethod
    def count(list1, x, y, t_set, row, column):

        """Sum up the mines nearby"""

        s = 0
        for i in range(x - 1, x + 2):
            if i < 0:
                continue
            elif i >= row:
                continue
            else:
                for j in range(y - 1, y + 2):
                    if j < 0:
                        continue
                    elif j >= column:
                        continue
                    elif abs(i - x) + abs(j - y) == 2:
                        continue
                    elif (i, j) in t_set:
                        continue
                    else:
                        if list1[i][j] == 1:
                            s += 1
                            if (i, j) == (x, y):
                                t_set.add((x, y))
                                continue
                            else:
                                t_set.add((i, j))
                                s += Game.count(list1, i, j, t_set, row, column)
        return s

    # ------------------------------------------------------------------------------------------------------------------
    # Core Function 2_1_2

    @staticmethod
    def adjust(list1, set1, t_set, row, column, bound):

        """Judge whether the 'mine-map' should be re-initialized"""

        m = 0
        for i in set1:
            x = i[0]
            y = i[1]
            temp = Game.count(list1, x, y, t_set, row, column)
            if temp > m:
                m = temp
        if m <= bound:
            return False
        else:
            return True

    # ------------------------------------------------------------------------------------------------------------------
    # Core Function 2_2_1_1

    @staticmethod
    def judge1(list2, x, y, row, column):

        """Judge whether to help or not"""

        for i in range(x - 1, x + 2):
            if i < 0:
                continue
            elif i >= row:
                continue
            else:
                for j in range(y - 1, y + 2):
                    if j < 0:
                        continue
                    elif j >= column:
                        continue
                    elif list2[i][j] == "":
                        return False
        return True

    # ------------------------------------------------------------------------------------------------------------------
    # Core Function 2_2_1_2

    @staticmethod
    def judge2(list1, list2, x, y, row, column, num):

        """Judge whether all mines nearby are 'checked'"""

        s = 0
        for i in range(x - 1, x + 2):
            if i < 0:
                continue
            elif i >= row:
                continue
            else:
                for j in range(y - 1, y + 2):
                    if j < 0:
                        continue
                    elif j >= column:
                        continue
                    elif list2[i][j] == "P":
                        if list1[i][j] == 1:
                            s += 1
                        else:
                            return False
        if s == num and num != 0:
            return True
        else:
            return False

    # ------------------------------------------------------------------------------------------------------------------
    # Core Function 2_2_2

    @staticmethod
    def helper(list1, list2, x, y, row, column, all_set):

        """Help the player a little bit"""

        num_set = set("0123456789")
        s_temp = 0
        for i in range(x - 1, x + 2):
            if i < 0:
                continue
            elif i >= row:
                continue
            else:
                for j in range(y - 1, y + 2):
                    if j < 0:
                        continue
                    elif j >= column:
                        continue
                    elif list2[i][j] in num_set:
                        num = int(list2[i][j])
                        if Game.judge1(list2, i, j, row, column) and Game.judge2(list1, list2, i, j, row, column, num):
                            s_temp += 1
                            break
                if s_temp > 0:
                    break
        if s_temp > 0:
            for i in range(x - 1, x + 2):
                if i < 0:
                    continue
                elif i >= row:
                    continue
                else:
                    for j in range(y - 1, y + 2):
                        if j < 0:
                            continue
                        elif j >= column:
                            continue
                        elif list2[i][j] in num_set:
                            num = int(list2[i][j])
                            if Game.judge2(list1, list2, i, j, row, column, num):
                                for i1 in range(i - 1, i + 2):
                                    if i1 < 0:
                                        continue
                                    elif i1 >= row:
                                        continue
                                    else:
                                        for j1 in range(j - 1, j + 2):
                                            if j1 < 0:
                                                continue
                                            elif j1 >= column:
                                                continue
                                            elif list2[i1][j1] == "":
                                                Game.dig_out(i1, j1, list1, list2, row, column, all_set)

    # ==================================================================================================================
    # Main Loop
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, pre_frame=None):

        if pre_frame is not None:
            pre_frame.master.destroy()

        # Frame
        tk.Frame.__init__(self)
        self.master.title("The MineSweeper Game")
        self.grid()

        self.master_frame = tk.LabelFrame()
        self.status_frame = tk.LabelFrame(self.master_frame)
        self.main_frame = tk.LabelFrame(self.master_frame)

        self.row_frame = tk.LabelFrame(self.master_frame)
        self.column_frame = tk.LabelFrame(self.master_frame)
        self.mine_frame = tk.LabelFrame(self.master_frame)

        self.status_frame.grid()
        self.main_frame.grid()

        # Menu
        menu_bar = tk.Menu(self)
        file_menu = tk.Menu(menu_bar)
        file_menu.add_command(label="Restart", command=self.replay)
        file_menu.add_command(label="Switch on/off the helper", command=self.switch_helper)
        file_menu.add_command(label="Quit", command=self.quit_game)

        menu_bar.add_cascade(label="File", menu=file_menu)
        self.master.config(menu=menu_bar)

        # ==============================================================================================================
        # Initialize
        # --------------------------------------------------------------------------------------------------------------
        # Row, Column, Left-mine Set(set1), Mine List(list1)
        self.row = self.column = self.mine_num = 0
        self.set1, self.list1 = set(), []

        # Max row/column
        self.m_max = 20

        self.master_frame.grid()

        def quit_set_mine():
            nonlocal self
            control = False
            try:
                self.mine_num = int(self.mine_text.get())
                if self.mine_num < self.s_min or self.mine_num > self.s_max:
                    self.mine_text.set("Please input an integer between {} and {}!!".format(self.s_min, self.s_max))
                else:
                    control = True
            except ValueError as err:
                self.mine_text.set(err)

            if control:
                self.mine_frame.grid_remove()

                # Set the rest properties
                a_bool = True
                tkm.showinfo(message="Initializing the map...")
                self.list1 = []
                self.set1 = set()
                while a_bool:
                    self.list1 = [[0 for i in range(self.column)] for j in range(self.row)]
                    self.set1 = set()
                    while len(self.set1) < self.mine_num:
                        self.set1.add((randint(0, self.row - 1), randint(0, self.column - 1)))
                    for temp in self.set1:
                        self.list1[temp[0]][temp[1]] = 1

                    t_set = set()
                    s1 = self.mine_num / (self.column * self.row)
                    s2 = sqrt(self.column * self.row)
                    if s1 < 0.2:
                        bound = 3 + s2 // 10
                    elif s1 < 0.3:
                        bound = 5 + s2 // 10
                    elif s1 < 0.45:
                        bound = s1 * 10 + s2 // 2
                    else:
                        bound = s1 * 10 + s2
                    a_bool = Game.adjust(self.list1, self.set1, t_set, self.row, self.column, bound)
                tkm.showinfo(message="Done!")

            # Varia2:
            self.helper_switch = 1
            self.condition = ""
            self.flag_num = self.mine_num
            self.correct = 0
            self.all_set = set()
            self.first = True
            self.status = "None"
            self.button_switch = False

            # dig_out list
            self.list2 = [["" for i in range(self.column)] for j in range(self.row)]

            # Set Action Buttons
            self.temp_button1 = tk.Button(self.status_frame, text="Dig", width=15, command=self.dig)
            self.temp_button2 = tk.Button(self.status_frame, text="Place a flag", width=15, command=self.check)
            self.temp_button3 = tk.Button(self.status_frame, text="Recycle a flag", width=15, command=self.recycle)
            self.temp_button1.grid(row=0, column=0)
            self.temp_button2.grid(row=0, column=1)
            self.temp_button3.grid(row=0, column=2)

            # Main Buttons
            self.button_lst = [[tk.Button(self.main_frame, width=3, height=1, command=self.set_function(i, j))
                                for j in range(self.column)] for i in range(self.row)]
            for i in range(self.row):
                for j in range(self.column):
                    self.button_lst[i][j].grid(row=i, column=j)

            # ===============================================================================================================
            # Show Various Information
            # ---------------------------------------------------------------------------------------------------------------
            # Show GUI
            self.show_gui()

            # Show Status
            self.status_var = tk.StringVar()
            self.status_var.set("Your current status: " + self.status)
            self.status_content = tk.Label(self.status_frame, textvariable=self.status_var)
            self.status_content.grid(row=1, column=0, columnspan=3)

            # Show Tips
            self.mine_num_label = tk.Label(self.status_frame, text="Total mines: {}".format(self.mine_num))
            self.flag_num_var = tk.IntVar()
            self.flag_num_var.set("Flags left: {}".format(self.flag_num))
            self.flag_num_label = tk.Label(self.status_frame, textvariable=self.flag_num_var)
            self.mine_num_label.grid(row=2, column=0, columnspan=3)
            self.flag_num_label.grid(row=3, column=0, columnspan=3)

            # Turn on the Buttons
            self.button_switch = True

            # Warning
            self.warning_var = tk.StringVar()
            self.warning_var.set("")
            self.warning_label = tk.Label(self.status_frame, textvariable=self.warning_var)
            self.warning_label.grid(row=5, column=0, columnspan=3)

        def quit_set_column():
            nonlocal self
            control = False
            try:
                self.column = int(self.column_text.get())
                if self.column < 2 or self.column > self.m_max:
                    self.column_text.set("Please input an integer between {} and {}!!".format(2, self.m_max))
                else:
                    control = True
            except ValueError as err:
                self.column_text.set(err)

            if control:
                self.column_frame.grid_remove()

                # Set mine

                # Varia1:
                s = self.row * self.column
                self.s_min = s // 10 if s >= 10 else 1
                self.s_max = s // 2

                # Quantity of mines:
                self.mine_text = tk.StringVar()
                self.mine_text.set("")
                mine_label = tk.Label(self.mine_frame, text="Please set the mine ({} ~ {})"
                                      .format(self.s_min, self.s_max))
                mine_entry = tk.Entry(self.mine_frame, textvariable=self.mine_text, width=40)
                mine_entry.bind("<Return>", lambda event: quit_set_mine())
                mine_button = tk.Button(self.mine_frame, text="Click me to set", bd=0, bg="PaleGreen", width=34,
                                        command=quit_set_mine)
                mine_label.grid()
                mine_entry.grid()
                mine_button.grid()
                self.mine_frame.grid()
                mine_entry.focus_set()

        def quit_set_row():
            nonlocal self
            control = False
            try:
                self.row = int(self.row_text.get())
                if self.row < 2 or self.row > self.m_max:
                    self.row_text.set("Please input an integer between {} and {}!!".format(2, self.m_max))
                else:
                    control = True
            except ValueError as err:
                self.row_text.set(err)

            if control:
                self.row_frame.grid_remove()

                # Set column

                self.column_text = tk.StringVar()
                self.column_text.set("")
                column_label = tk.Label(self.column_frame, text="Please set the column (2 ~ {})".format(self.m_max))
                column_entry = tk.Entry(self.column_frame, textvariable=self.column_text, width=40)
                column_entry.bind("<Return>", lambda event: quit_set_column())
                column_button = tk.Button(self.column_frame, text="Click me to set", bd=0, bg="PaleGreen", width=34,
                                          command=quit_set_column)
                column_label.grid()
                column_entry.grid()
                column_button.grid()
                self.column_frame.grid()
                column_entry.focus_set()

        # row:
        self.row_text = tk.StringVar()
        self.row_text.set("")
        row_label = tk.Label(self.row_frame, text="Please set the row (2 ~ {})".format(self.m_max))
        row_entry = tk.Entry(self.row_frame, textvariable=self.row_text, width=40)
        row_entry.bind("<Return>", lambda event: quit_set_row())
        row_button = tk.Button(self.row_frame, text="Click me to set", bd=0, bg="PaleGreen", width=34,
                               command=quit_set_row)
        row_label.grid()
        row_entry.grid()
        row_button.grid()
        self.row_frame.grid()
        row_entry.focus_set()

    def dig(self):
        if self.button_switch:
            self.status = "Dig"
            self.status_var.set("Your current status: " + self.status)

    def dig_action(self, x, y):
        if self.list2[x][y] == "P":
            self.warning_var.set("There's a flag here!!")
        elif self.list1[x][y] == 1:
            if self.first:
                self.first = False
                self.list1[x][y] = 0
                self.set1.remove((x, y))
                while len(self.set1) < self.mine_num:
                    x1 = randint(0, self.row - 1)
                    y1 = randint(0, self.column - 1)
                    if (x1, y1) != (x, y):
                        self.set1.add((randint(0, self.row - 1), randint(0, self.column - 1)))
                for temp in self.set1:
                    self.list1[temp[0]][temp[1]] = 1
                Game.dig_out(x, y, self.list1, self.list2, self.row, self.column, self.all_set)

                self.show_gui()
                self.warning_var.set("Digging...")

            else:
                self.warning_var.set("Boom!!!")
                self.list2[x][y] = "#"

                for i in range(self.row):
                    for j in range(self.column):
                        if (self.list1[i][j] == 1) and (self.list2[i][j] == ""):
                            self.list2[i][j] = "*"
                        elif (self.list1[i][j] == 0) and (self.list2[i][j] == "P"):
                            self.list2[i][j] = "?"

                self.warning_var.set("Well... You lose!!")
                self.condition = "lose"
                self.show_gui()
                self.run_result()

        elif self.list2[x][y] != "":
            self.warning_var.set("You can't dig here!!")
        else:
            Game.dig_out(x, y, self.list1, self.list2, self.row, self.column, self.all_set)
            self.show_gui()
            self.warning_var.set("Digging...")

    def check(self):
        if self.button_switch:
            self.status = "Place a flag"
            self.status_var.set("Your current status: " + self.status)

    def check_action(self, x, y):
        if self.flag_num <= 0:
            self.warning_var.set("You don't have any flag now!!")
        else:
            if self.list2[x][y] == "P":
                self.warning_var.set("There's a flag here already!!".format(x + 1, y + 1))
            elif self.list2[x][y] != "":
                self.warning_var.set("You can't place a flag here!!")
            else:
                self.first = False
                self.list2[x][y] = "P"
                self.flag_num -= 1
                self.flag_num_var.set("Flags left: {}".format(self.flag_num))
                self.warning_var.set("Placing flags...")
                self.all_set.add((x, y))

                if self.helper_switch == 1:
                    Game.helper(self.list1, self.list2, x, y, self.row, self.column, self.all_set)

                self.show_gui()

        if self.list1[x][y] == 1:
            self.correct += 1
        if self.correct == self.mine_num:
            self.condition = "win"
            self.warning_var.set("Congratulation! You win!!")
            self.run_result()

        if self.flag_num == 0:
            if self.condition != "win":
                self.warning_var.set("Oops... No more flags left...")
                self.status = "None"
                self.status_var.set("Your current status: " + self.status)

    def recycle(self):
        if self.button_switch:
            self.status = "Recycle a flag"
            self.status_var.set("Your current status: " + self.status)

    def recycle_action(self, x, y):
        if self.list2[x][y] != "P":
            self.warning_var.set("No flag here!!".format(x + 1, y + 1))
        else:
            self.first = False
            self.flag_num += 1
            self.flag_num_var.set("Flags left: {}".format(self.flag_num))
            self.list2[x][y] = ""
            if self.list1[x][y] == 1:
                self.correct -= 1

            self.all_set.remove((x, y))

            retry_temp = 0
            for i in range(x - 1, x + 2):
                if i < 0:
                    continue
                elif i >= self.row:
                    continue
                else:
                    for j in range(y - 1, y + 2):
                        if j < 0:
                            continue
                        elif j >= self.column:
                            continue
                        elif self.list2[i][j] == "0":
                            Game.dig_out(x, y, self.list1, self.list2, self.row, self.column, self.all_set)
                            retry_temp = 1
                            break
                        else:
                            continue
                    if retry_temp == 1:
                        break
            self.warning_var.set("Recycling flags...")
            self.show_gui()

            if self.flag_num == self.mine_num:
                self.warning_var.set("Well, you've got all your flags!!")
                self.status = "None"
                self.status_var.set("Your current status: " + self.status)

    def set_function(self, x, y):

        def sub_function():
            self.button_command(x, y)

        return sub_function

    def button_command(self, x, y):
        if self.button_switch:
            if self.status == "None":
                self.warning_var.set("Please set your action first!")

            elif self.status == "Recycle a flag":
                self.recycle_action(x, y)

            elif self.status == "Place a flag":
                self.check_action(x, y)

            elif self.status == "Dig":
                self.dig_action(x, y)
        else:
            pass

    def run_result(self):
        replay_button = tk.Button(self, text="Click me to play again!", width=30, command=self.replay)
        replay_button.grid(row=6, column=0, columnspan=3)
        self.button_switch = False

    def replay(self):
        Game(self).mainloop()

    def switch_helper(self):
        self.helper_switch = 1 - self.helper_switch

    def show_gui(self):
        for i in range(self.row):
            for j in range(self.column):
                temp_char = self.list2[i][j]
                temp_button = self.button_lst[i][j]

                if temp_char == "P":
                    temp_button["bg"] = "light green"
                elif temp_char == "":
                    temp_button["bg"] = "white smoke"
                elif self.condition != "lose":
                    temp_button["relief"] = "groove"
                elif self.condition == "lose":
                    if temp_char == "#":
                        temp_button["bg"] = "red"
                        temp_button["fg"] = "white"
                    if temp_char == "*":
                        temp_button["bg"] = "rosy brown"
                        temp_button["fg"] = "white"
                    elif temp_char == "?":
                        temp_button["bg"] = "yellow"

                self.button_lst[i][j]["text"] = self.list2[i][j]

    def quit_game(self):
        self.master.destroy()

Game().mainloop()
