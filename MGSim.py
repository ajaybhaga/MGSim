#!/usr/bin/env python
#----------------------------------------------------------------------------
# Name:         MGSim.py
# Purpose:      Munvo Message Gateway Simulator
#
# Author:       Ajay Bhaga (ajay.bhaga@munvo.com)
#
# Created:      25-May-2020
# Copyright:    (c) 2020 by Munvo
#----------------------------------------------------------------------------

'''
This program is a Message Gateway Simulator which produces referential models
for iterative training.  This simulator provides a workbench to tweak models
to meet preferred goals.  The generates patterns that can be used to deploy
on Message Gateway to validate accuracy of results.

- wxPython to manage cross-platform window management
- PostgreSQL server to manage internal system metadata
- TensorFlow for deep reinforcement learning
- wx.OGL for 2D graphic management
- GLUT/OpenGL for any optional 3d rendering (model visualizaton)
- C++ for Message Gateway Simulator core code

'''

import numpy as np
import random
import wx
import wx.lib.inspection
import wx.lib.mixins.inspection
import wx.grid
import wx.html
import wx.aui as aui
import sys, os

from six import BytesIO

try:
    from wx import glcanvas
    haveGLCanvas = True
except ImportError:
    haveGLCanvas = False

try:
    # The Python OpenGL package can be found at
    # http://PyOpenGL.sourceforge.net/
    from OpenGL.GL import *
    from OpenGL.GLUT import *
    haveOpenGL = True
except ImportError:
    haveOpenGL = False


from env.mgsim_env import MGSimEnv
from learning.rl_world import RLWorld
from util.arg_parser import ArgParser
from util.logger import Logger
import util.mpi_util as MPIUtil
import util.util as Util

ID_CreateTree = wx.NewIdRef()
ID_CreateGrid = wx.NewIdRef()
ID_CreateText = wx.NewIdRef()
ID_CreateHTML = wx.NewIdRef()
ID_CreateSizeReport = wx.NewIdRef()
ID_GridContent = wx.NewIdRef()
ID_TextContent = wx.NewIdRef()
ID_TreeContent = wx.NewIdRef()
ID_HTMLContent = wx.NewIdRef()
ID_SizeReportContent = wx.NewIdRef()
ID_CreatePerspective = wx.NewIdRef()
ID_CopyPerspective = wx.NewIdRef()

ID_TransparentHint = wx.NewIdRef()
ID_VenetianBlindsHint = wx.NewIdRef()
ID_RectangleHint = wx.NewIdRef()
ID_NoHint = wx.NewIdRef()
ID_HintFade = wx.NewIdRef()
ID_AllowFloating = wx.NewIdRef()
ID_NoVenetianFade = wx.NewIdRef()
ID_TransparentDrag = wx.NewIdRef()
ID_AllowActivePane = wx.NewIdRef()
ID_NoGradient = wx.NewIdRef()
ID_VerticalGradient = wx.NewIdRef()
ID_HorizontalGradient = wx.NewIdRef()

ID_Settings = wx.NewIdRef()
ID_About = wx.NewIdRef()

ID_NavData = wx.NewIdRef()
ID_NavModel = wx.NewIdRef()
ID_NavExpr = wx.NewIdRef()
ID_NavResults = wx.NewIdRef()

ID_LoadModel = wx.NewIdRef()
ID_SaveModel = wx.NewIdRef()
ID_ExportPattern = wx.NewIdRef()


# This reserves count IDs and returns a list of WindowIDRef objects
ID_FirstPerspective = wx.NewIdRef(100)


# Dimensions of the window we are drawing into.
win_width = 1024
win_height = int(win_width * 9.0 / 16.0)
reshaping = False

# anim
fps = 60
update_timestep = 1.0 / fps
display_anim_time = int(1000 * update_timestep)
animating = True

playback_speed = 1
playback_delta = 0.05

# FPS counter
prev_time = 0
updates_per_sec = 0

args = []
world = None

assertMode = wx.APP_ASSERT_DIALOG
##assertMode = wx.APP_ASSERT_EXCEPTION

#----------------------------------------------------------------------------

class Log:
    def WriteText(self, text):
        if text[-1:] == '\n':
            text = text[:-1]
        wx.LogMessage(text)
    write = WriteText


import wx
import wx.lib.layoutf as layoutf


#---------------------------------------------------------------------------

class TestDialog(wx.Dialog):
    def __init__(
            self, parent, id, title, size=wx.DefaultSize, pos=wx.DefaultPosition,
            style=wx.DEFAULT_DIALOG_STYLE, name='dialog'
    ):

        # Instead of calling wx.Dialog.__init__ we precreate the dialog
        # so we can set an extra style that must be set before
        # creation, and then we create the GUI object using the Create
        # method.
        wx.Dialog.__init__(self)
        self.SetExtraStyle(wx.DIALOG_EX_CONTEXTHELP)
        self.Create(parent, id, title, pos, size, style, name)

        # Now continue with the normal construction of the dialog
        # contents
        sizer = wx.BoxSizer(wx.VERTICAL)

        label = wx.StaticText(self, -1, "This is a wx.Dialog")
        label.SetHelpText("This is the help text for the label")
        sizer.Add(label, 0, wx.ALIGN_CENTRE|wx.ALL, 5)

        box = wx.BoxSizer(wx.HORIZONTAL)

        label = wx.StaticText(self, -1, "Field #1:")
        label.SetHelpText("This is the help text for the label")
        box.Add(label, 0, wx.ALIGN_CENTRE|wx.ALL, 5)

        text = wx.TextCtrl(self, -1, "", size=(80,-1))
        text.SetHelpText("Here's some help text for field #1")
        box.Add(text, 1, wx.ALIGN_CENTRE|wx.ALL, 5)

        sizer.Add(box, 0, wx.EXPAND|wx.ALL, 5)

        box = wx.BoxSizer(wx.HORIZONTAL)

        label = wx.StaticText(self, -1, "Field #2:")
        label.SetHelpText("This is the help text for the label")
        box.Add(label, 0, wx.ALIGN_CENTRE|wx.ALL, 5)

        text = wx.TextCtrl(self, -1, "", size=(80,-1))
        text.SetHelpText("Here's some help text for field #2")
        box.Add(text, 1, wx.ALIGN_CENTRE|wx.ALL, 5)

        sizer.Add(box, 0, wx.EXPAND|wx.ALL, 5)

        line = wx.StaticLine(self, -1, size=(20,-1), style=wx.LI_HORIZONTAL)

        sizer.Add(line, 0, wx.EXPAND|wx.RIGHT|wx.TOP, 5)

        btnsizer = wx.StdDialogButtonSizer()

        if wx.Platform != "__WXMSW__":
            btn = wx.ContextHelpButton(self)
            btnsizer.AddButton(btn)

        btn = wx.Button(self, wx.ID_OK)
        btn.SetHelpText("The OK button completes the dialog")
        btn.SetDefault()
        btnsizer.AddButton(btn)

        btn = wx.Button(self, wx.ID_CANCEL)
        btn.SetHelpText("The Cancel button cancels the dialog. (Cool, huh?)")
        btnsizer.AddButton(btn)
        btnsizer.Realize()

        sizer.Add(btnsizer, 0, wx.ALL, 5)

        self.SetSizer(sizer)
        sizer.Fit(self)


#---------------------------------------------------------------------------

class PyAUIFrame(wx.Frame):

    def __init__(self, parent, id=-1, title="", pos=wx.DefaultPosition,
                 size=wx.DefaultSize, style=wx.DEFAULT_FRAME_STYLE |
                                            wx.SUNKEN_BORDER |
                                            wx.CLIP_CHILDREN):

        wx.Frame.__init__(self, parent, id, title, pos, size, style)

        # tell FrameManager to manage this frame
        self._mgr = aui.AuiManager()
        self._mgr.SetManagedWindow(self)

        self._perspectives = []
        self.n = 0
        self.x = 0

        #self.SetIcon(GetMondrianIcon())

        # create menu
        mb = wx.MenuBar()

        file_menu = wx.Menu()
        file_menu.Append(ID_LoadModel, "Load Model")
        file_menu.Append(ID_SaveModel, "Save Model")
        file_menu.Append(ID_ExportPattern, "Export Pattern")
        file_menu.Append(wx.ID_EXIT, "Exit")

        #view_menu = wx.Menu()
        #view_menu.Append(ID_CreateText, "Create Text Control")
        #view_menu.Append(ID_CreateHTML, "Create HTML Control")
        #view_menu.Append(ID_CreateTree, "Create Tree")
        #view_menu.Append(ID_CreateGrid, "Create Grid")
        #view_menu.Append(ID_CreateSizeReport, "Create Size Reporter")
        #view_menu.AppendSeparator()
        #view_menu.Append(ID_GridContent, "Use a Grid for the Content Pane")
        #view_menu.Append(ID_TextContent, "Use a Text Control for the Content Pane")
        #view_menu.Append(ID_HTMLContent, "Use an HTML Control for the Content Pane")
        #view_menu.Append(ID_TreeContent, "Use a Tree Control for the Content Pane")
        #view_menu.Append(ID_SizeReportContent, "Use a Size Reporter for the Content Pane")

        options_menu = wx.Menu()
        options_menu.AppendRadioItem(ID_TransparentHint, "Transparent Hint")
        options_menu.AppendRadioItem(ID_VenetianBlindsHint, "Venetian Blinds Hint")
        options_menu.AppendRadioItem(ID_RectangleHint, "Rectangle Hint")
        options_menu.AppendRadioItem(ID_NoHint, "No Hint")
        options_menu.AppendSeparator();
        options_menu.AppendCheckItem(ID_HintFade, "Hint Fade-in")
        options_menu.AppendCheckItem(ID_AllowFloating, "Allow Floating")
        options_menu.AppendCheckItem(ID_NoVenetianFade, "Disable Venetian Blinds Hint Fade-in")
        options_menu.AppendCheckItem(ID_TransparentDrag, "Transparent Drag")
        options_menu.AppendCheckItem(ID_AllowActivePane, "Allow Active Pane")
        options_menu.AppendSeparator();
        options_menu.AppendRadioItem(ID_NoGradient, "No Caption Gradient")
        options_menu.AppendRadioItem(ID_VerticalGradient, "Vertical Caption Gradient")
        options_menu.AppendRadioItem(ID_HorizontalGradient, "Horizontal Caption Gradient")
        options_menu.AppendSeparator();
        options_menu.Append(ID_Settings, "Settings Pane")

        #self._perspectives_menu = wx.Menu()
        #self._perspectives_menu.Append(ID_CreatePerspective, "Create Perspective")
        #self._perspectives_menu.Append(ID_CopyPerspective, "Copy Perspective Data To Clipboard")
        #self._perspectives_menu.AppendSeparator()
        #self._perspectives_menu.Append(ID_FirstPerspective[0], "Default Startup")
        #self._perspectives_menu.Append(ID_FirstPerspective[1], "All Panes")
        #self._perspectives_menu.Append(ID_FirstPerspective[2], "Vertical Toolbar")

        help_menu = wx.Menu()
        help_menu.Append(ID_About, "About...")

        mb.Append(file_menu, "File")
        #mb.Append(view_menu, "View")
        #mb.Append(self._perspectives_menu, "Perspectives")
        mb.Append(options_menu, "Options")
        mb.Append(help_menu, "Help")

        self.SetMenuBar(mb)

        self.statusbar = self.CreateStatusBar(2, wx.STB_SIZEGRIP)
        self.statusbar.SetStatusWidths([-2, -3])
        self.statusbar.SetStatusText("Ready", 0)
        self.statusbar.SetStatusText("Please load model.", 1)

        # min size for the frame itself isn't completely done.
        # see the end up FrameManager::Update() for the test
        # code. For now, just hard code a frame minimum size
        self.SetMinSize(wx.Size(400, 300))

        # create some toolbars
        filenew_bmp1 = wx.ArtProvider.GetBitmap(wx.ART_NEW, wx.ART_TOOLBAR, wx.Size(16, 16))
        fileopen_bmp1 = wx.ArtProvider.GetBitmap(wx.ART_FILE_OPEN, wx.ART_TOOLBAR, wx.Size(16, 16))
        filesave_bmp1 = wx.ArtProvider.GetBitmap(wx.ART_FILE_SAVE, wx.ART_TOOLBAR, wx.Size(16, 16))
        patternexp_bmp1 = wx.ArtProvider.GetBitmap(wx.ART_FILE_SAVE_AS, wx.ART_TOOLBAR, wx.Size(16, 16))
        disk_bmp1 = wx.ArtProvider.GetBitmap(wx.ART_HARDDISK, wx.ART_TOOLBAR, wx.Size(16, 16))
        model_bmp1 = wx.ArtProvider.GetBitmap(wx.ART_LIST_VIEW, wx.ART_TOOLBAR, wx.Size(16, 16))
        result_bmp1 = wx.ArtProvider.GetBitmap(wx.ART_REPORT_VIEW, wx.ART_TOOLBAR, wx.Size(16, 16))
        expr_bmp1 = wx.ArtProvider.GetBitmap(wx.ART_EXECUTABLE_FILE, wx.ART_TOOLBAR, wx.Size(16, 16))

        tb3 = wx.ToolBar(self, -1, wx.DefaultPosition, wx.DefaultSize,
                         wx.TB_FLAT | wx.TB_HORZ_TEXT)
        tb3.SetToolBitmapSize(wx.Size(16,16))

        tb3.AddTool(101, "New Simulation", filenew_bmp1)
        self.Bind(wx.EVT_TOOL, self.OnToolClick, id=101)
        self.Bind(wx.EVT_TOOL_RCLICKED, self.OnToolRClick, id=101)

        tb3.AddTool(102, "Load Model", fileopen_bmp1)
        self.Bind(wx.EVT_TOOL, self.OnToolClick, id=102)
        self.Bind(wx.EVT_TOOL_RCLICKED, self.OnToolRClick, id=102)

        tb3.AddTool(103, "Save Model", filesave_bmp1)
        self.Bind(wx.EVT_TOOL, self.OnToolClick, id=103)
        self.Bind(wx.EVT_TOOL_RCLICKED, self.OnToolRClick, id=103)

        tb3.AddTool(104, "Export Pattern", patternexp_bmp1)
        self.Bind(wx.EVT_TOOL, self.OnToolClick, id=104)
        self.Bind(wx.EVT_TOOL_RCLICKED, self.OnToolRClick, id=104)

#        tb3.AddSeparator()
#        tb3.AddTool(101, "Test", tb3_bmp1)
#        tb3.AddTool(101, "Test", tb3_bmp1)
        tb3.Realize()


        tb4 = wx.ToolBar(self, -1, wx.DefaultPosition, wx.DefaultSize,
                         wx.TB_FLAT | wx.TB_NODIVIDER | wx.TB_HORZ_TEXT | wx.TB_VERTICAL)
        tb4.SetToolBitmapSize(wx.Size(16,16))
        tb4.AddTool(105, "Data", disk_bmp1)
        self.Bind(wx.EVT_TOOL, self.OnToolClick, id=105)
        self.Bind(wx.EVT_TOOL_RCLICKED, self.OnToolRClick, id=105)

        tb4.AddTool(106, "Model", model_bmp1)
        self.Bind(wx.EVT_TOOL, self.OnToolClick, id=106)
        self.Bind(wx.EVT_TOOL_RCLICKED, self.OnToolRClick, id=106)

        tb4.AddTool(107, "Experiment", expr_bmp1)
        self.Bind(wx.EVT_TOOL, self.OnToolClick, id=107)
        self.Bind(wx.EVT_TOOL_RCLICKED, self.OnToolRClick, id=107)

        tb4.AddTool(108, "Results", result_bmp1)
        self.Bind(wx.EVT_TOOL, self.OnToolClick, id=108)
        self.Bind(wx.EVT_TOOL_RCLICKED, self.OnToolRClick, id=108)

        tb4.Realize()

        '''
        tb5 = wx.ToolBar(self, -1, wx.DefaultPosition, wx.DefaultSize,
                         wx.TB_FLAT | wx.TB_NODIVIDER | wx.TB_VERTICAL)
        tb5.SetToolBitmapSize(wx.Size(48, 48))
        tb5.AddTool(101, "Test", wx.ArtProvider.GetBitmap(wx.ART_ERROR))
        tb5.AddSeparator()
        tb5.AddTool(102, "Test", wx.ArtProvider.GetBitmap(wx.ART_QUESTION))
        tb5.AddTool(103, "Test", wx.ArtProvider.GetBitmap(wx.ART_INFORMATION))
        tb5.AddTool(103, "Test", wx.ArtProvider.GetBitmap(wx.ART_WARNING))
        tb5.AddTool(103, "Test", wx.ArtProvider.GetBitmap(wx.ART_MISSING_IMAGE))
        tb5.Realize()
        '''

        # add a bunch of panes
        self._mgr.AddPane(self.CreateTreeCtrl(), aui.AuiPaneInfo().
                          Name("datatree1").Caption("Data Tree Pane").
                          Left().Layer(1).Position(1).CloseButton(True).MaximizeButton(True))

        self._mgr.AddPane(self.CreateStatusTextCtrl(), aui.AuiPaneInfo().
                          Name("status1").Caption("Status").
                          Bottom().Layer(1).Position(1).CloseButton(True).MaximizeButton(True))

        self._mgr.AddPane(self.CreateSizeReportCtrl(), aui.AuiPaneInfo().
                          Name("test11").Caption("Fixed Pane").
                          Bottom().Layer(1).Position(2).Fixed().CloseButton(True).MaximizeButton(True))

        self._mgr.AddPane(SettingsPanel(self, self), aui.AuiPaneInfo().
                          Name("settings").Caption("Dock Manager Settings").
                          Dockable(False).Float().Hide().CloseButton(True).MaximizeButton(True))

        # add the toolbars to the manager
        self._mgr.AddPane(tb3, aui.AuiPaneInfo().
                          Name("tb3").Caption("Simulation Toolbar").
                          ToolbarPane().Top().
                          LeftDockable(False).RightDockable(False))

        self._mgr.AddPane(tb4, aui.AuiPaneInfo().
                          Name("tb4").Caption("Navigator Toolbar").
                          ToolbarPane().Left().GripperTop().
                          TopDockable(False).BottomDockable(False))

        # make some default perspectives
        perspective_all = self._mgr.SavePerspective()

        all_panes = self._mgr.GetAllPanes()

        for ii in range(len(all_panes)):
            if not all_panes[ii].IsToolbar():
                all_panes[ii].Hide()

        self._mgr.GetPane("tb1").Hide()
        self._mgr.GetPane("tb5").Hide()
        self._mgr.GetPane("datatree1").Show().Left().Layer(0).Row(0).Position(0)
        self._mgr.GetPane("status1").Show().Bottom().Layer(0).Row(0).Position(0)
        self._mgr.GetPane("html_content").Show()

        perspective_default = self._mgr.SavePerspective()

        for ii in range(len(all_panes)):
            if not all_panes[ii].IsToolbar():
                all_panes[ii].Hide()

        self._mgr.GetPane("tb1").Hide()
        self._mgr.GetPane("tb5").Hide()
        self._mgr.GetPane("grid_content").Show()
        self._mgr.GetPane("datatree1").Show().Left().Layer(0).Row(0).Position(0)
        self._mgr.GetPane("status1").Show().Bottom().Layer(0).Row(0).Position(0)
        self._mgr.GetPane("html_content").Show()

        perspective_vert = self._mgr.SavePerspective()

        self._perspectives.append(perspective_default)
        self._perspectives.append(perspective_all)
        self._perspectives.append(perspective_vert)

        self._mgr.GetPane("grid_content").Hide()

        # "commit" all changes made to FrameManager
        self._mgr.Update()

        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_CLOSE, self.OnClose)

        # Show How To Use The Closing Panes Event
        self.Bind(aui.EVT_AUI_PANE_CLOSE, self.OnPaneClose)

        self.Bind(wx.EVT_MENU, self.OnCreateTree, id=ID_CreateTree)
        self.Bind(wx.EVT_MENU, self.OnCreateGrid, id=ID_CreateGrid)
        self.Bind(wx.EVT_MENU, self.OnCreateText, id=ID_CreateText)
        self.Bind(wx.EVT_MENU, self.OnCreateHTML, id=ID_CreateHTML)
        self.Bind(wx.EVT_MENU, self.OnCreateSizeReport, id=ID_CreateSizeReport)
        self.Bind(wx.EVT_MENU, self.OnCreatePerspective, id=ID_CreatePerspective)
        self.Bind(wx.EVT_MENU, self.OnCopyPerspective, id=ID_CopyPerspective)

        self.Bind(wx.EVT_MENU, self.OnNavData, id=ID_NavData)
        self.Bind(wx.EVT_MENU, self.OnNavModel, id=ID_NavModel)
        self.Bind(wx.EVT_MENU, self.OnNavExpr, id=ID_NavExpr)
        self.Bind(wx.EVT_MENU, self.OnNavResults, id=ID_NavResults)

        self.Bind(wx.EVT_MENU, self.OnManagerFlag, id=ID_AllowFloating)
        self.Bind(wx.EVT_MENU, self.OnManagerFlag, id=ID_TransparentHint)
        self.Bind(wx.EVT_MENU, self.OnManagerFlag, id=ID_VenetianBlindsHint)
        self.Bind(wx.EVT_MENU, self.OnManagerFlag, id=ID_RectangleHint)
        self.Bind(wx.EVT_MENU, self.OnManagerFlag, id=ID_NoHint)
        self.Bind(wx.EVT_MENU, self.OnManagerFlag, id=ID_HintFade)
        self.Bind(wx.EVT_MENU, self.OnManagerFlag, id=ID_NoVenetianFade)
        self.Bind(wx.EVT_MENU, self.OnManagerFlag, id=ID_TransparentDrag)
        self.Bind(wx.EVT_MENU, self.OnManagerFlag, id=ID_AllowActivePane)

        self.Bind(wx.EVT_MENU, self.OnGradient, id=ID_NoGradient)
        self.Bind(wx.EVT_MENU, self.OnGradient, id=ID_VerticalGradient)
        self.Bind(wx.EVT_MENU, self.OnGradient, id=ID_HorizontalGradient)
        self.Bind(wx.EVT_MENU, self.OnSettings, id=ID_Settings)
        self.Bind(wx.EVT_MENU, self.OnChangeContentPane, id=ID_GridContent)
        self.Bind(wx.EVT_MENU, self.OnChangeContentPane, id=ID_TreeContent)
        self.Bind(wx.EVT_MENU, self.OnChangeContentPane, id=ID_TextContent)
        self.Bind(wx.EVT_MENU, self.OnChangeContentPane, id=ID_SizeReportContent)
        self.Bind(wx.EVT_MENU, self.OnChangeContentPane, id=ID_HTMLContent)
        self.Bind(wx.EVT_MENU, self.OnExit, id=wx.ID_EXIT)
        self.Bind(wx.EVT_MENU, self.OnAbout, id=ID_About)

        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUI, id=ID_TransparentHint)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUI, id=ID_VenetianBlindsHint)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUI, id=ID_RectangleHint)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUI, id=ID_NoHint)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUI, id=ID_HintFade)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUI, id=ID_AllowFloating)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUI, id=ID_NoVenetianFade)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUI, id=ID_TransparentDrag)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUI, id=ID_AllowActivePane)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUI, id=ID_NoGradient)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUI, id=ID_VerticalGradient)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUI, id=ID_HorizontalGradient)


        self.Bind(wx.EVT_MENU_RANGE, self.OnRestorePerspective,
                  id=ID_FirstPerspective[0], id2=ID_FirstPerspective[-1])





    def OnTool101(self):
        Logger.print('Tool 101');
        dlg = TestDialog(self, -1, "New Simulation", size=(350, 200),
                             style=wx.DEFAULT_DIALOG_STYLE)
        dlg.ShowWindowModal()

    def OnTool102(self):
        Logger.print('Tool 102');
        dlg = TestDialog(self, -1, "Load Model", size=(350, 200),
                         style=wx.DEFAULT_DIALOG_STYLE)
        dlg.ShowWindowModal()

    def OnTool103(self):
        Logger.print('Tool 103');
        dlg = TestDialog(self, -1, "Save Model", size=(350, 200),
                         style=wx.DEFAULT_DIALOG_STYLE)
        dlg.ShowWindowModal()

    def OnTool104(self):
        Logger.print('Tool 104');
        dlg = TestDialog(self, -1, "Export Pattern", size=(350, 200),
                         style=wx.DEFAULT_DIALOG_STYLE)
        dlg.ShowWindowModal()

    def OnTool105(self):
        Logger.print('Navigate Data');
        # Update center content

    def OnTool106(self):
        Logger.print('Navigate Model');
        # Update center content

    def OnTool107(self):
        Logger.print('Navigate Experiment');
        # Update center content

    def OnTool108(self):
        Logger.print('Navigate Results');
        # Update center content

    def OnToolClick(self, event):
        Logger.print("tool %s clicked\n" % event.GetId())
        #tb = self.GetToolBar()
        tb = event.GetEventObject()
        #tb.EnableTool(10, not tb.GetToolEnabled(10))

        # New Simulation
        if (event.GetId() == 101):
            self.OnTool101()

        # Load Model
        if (event.GetId() == 102):
            self.OnTool102()

        # Save Model
        if (event.GetId() == 103):
            self.OnTool103()

        # Export Pattern
        if (event.GetId() == 104):
            self.OnTool104()

        # Navigate Data
        if (event.GetId() == 105):
            self.OnTool105()

        # Navigate Model
        if (event.GetId() == 106):
            self.OnTool106()

        # Navigate Experiment
        if (event.GetId() == 107):
            self.OnTool107()

        # Navigate Results
        if (event.GetId() == 108):
            self.OnTool108()


    def OnToolRClick(self, event):
        Logger.print("tool %s right-clicked\n" % event.GetId())

    def OnPaneClose(self, event):

        Logger.print('OnPaneClose: %s' % event)

        caption = event.GetPane().caption

        if caption in ["Tree Pane", "Dock Manager Settings", "Fixed Pane"]:
            msg = "Are You Sure You Want To Close This Pane?"
            dlg = wx.MessageDialog(self, msg, "AUI Question",
                                   wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION)

            if dlg.ShowModal() in [wx.ID_NO, wx.ID_CANCEL]:
                event.Veto()
            dlg.Destroy()


    def OnClose(self, event):
        Logger.print('OnClose: %s' % event)

        global world

        Logger.print('Shutting down...')
        world.shutdown()

        self._mgr.UnInit()
        del self._mgr
        self.Destroy()
        sys.exit(0)


    def OnExit(self, event):
        Logger.print('OnExit: %s' % event)
        self.Close()

    def OnAbout(self, event):

        msg = "Munvo Message Gateway: Simulator\n" + \
              "Message Gateway Simulator produces referential models\n" + \
              "for iterative training. This simulator provides a workbench to tweak models\n" + \
              "to meet preferred goals. The generates patterns that can be used to deploy\n" + \
              "on Message Gateway to validate accuracy of results.\n" + \
              "(c) Copyright 2020, Munvo";
        dlg = wx.MessageDialog(self, msg, "About Munvo Message Gateway Simulator",
                               wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()


    def GetDockArt(self):

        return self._mgr.GetArtProvider()


    def DoUpdate(self):

        self._mgr.Update()


    def OnEraseBackground(self, event):

        event.Skip()


    def OnSize(self, event):

        event.Skip()


    def OnSettings(self, event):

        # show the settings pane, and float it
        floating_pane = self._mgr.GetPane("settings").Float().Show()

        if floating_pane.floating_pos == wx.DefaultPosition:
            floating_pane.FloatingPosition(self.GetStartPosition())

        self._mgr.Update()


    def OnGradient(self, event):

        gradient = 0

        if event.GetId() == ID_NoGradient:
            gradient = aui.AUI_GRADIENT_NONE
        elif event.GetId() == ID_VerticalGradient:
            gradient = aui.AUI_GRADIENT_VERTICAL
        elif event.GetId() == ID_HorizontalGradient:
            gradient = aui.AUI_GRADIENT_HORIZONTAL

        self._mgr.GetArtProvider().SetMetric(aui.AUI_DOCKART_GRADIENT_TYPE, gradient)
        self._mgr.Update()


    def OnManagerFlag(self, event):

        flag = 0
        eid = event.GetId()

        if eid in [ ID_TransparentHint, ID_VenetianBlindsHint, ID_RectangleHint, ID_NoHint ]:
            flags = self._mgr.GetFlags()
            flags &= ~aui.AUI_MGR_TRANSPARENT_HINT
            flags &= ~aui.AUI_MGR_VENETIAN_BLINDS_HINT
            flags &= ~aui.AUI_MGR_RECTANGLE_HINT
            self._mgr.SetFlags(flags)

        if eid == ID_AllowFloating:
            flag = aui.AUI_MGR_ALLOW_FLOATING
        elif eid == ID_TransparentDrag:
            flag = aui.AUI_MGR_TRANSPARENT_DRAG
        elif eid == ID_HintFade:
            flag = aui.AUI_MGR_HINT_FADE
        elif eid == ID_NoVenetianFade:
            flag = aui.AUI_MGR_NO_VENETIAN_BLINDS_FADE
        elif eid == ID_AllowActivePane:
            flag = aui.AUI_MGR_ALLOW_ACTIVE_PANE
        elif eid == ID_TransparentHint:
            flag = aui.AUI_MGR_TRANSPARENT_HINT
        elif eid == ID_VenetianBlindsHint:
            flag = aui.AUI_MGR_VENETIAN_BLINDS_HINT
        elif eid == ID_RectangleHint:
            flag = aui.AUI_MGR_RECTANGLE_HINT

        self._mgr.SetFlags(self._mgr.GetFlags() ^ flag)


    def OnUpdateUI(self, event):

        flags = self._mgr.GetFlags()
        eid = event.GetId()

        if eid == ID_NoGradient:
            event.Check(self._mgr.GetArtProvider().GetMetric(aui.AUI_DOCKART_GRADIENT_TYPE) == aui.AUI_GRADIENT_NONE)

        elif eid == ID_VerticalGradient:
            event.Check(self._mgr.GetArtProvider().GetMetric(aui.AUI_DOCKART_GRADIENT_TYPE) == aui.AUI_GRADIENT_VERTICAL)

        elif eid == ID_HorizontalGradient:
            event.Check(self._mgr.GetArtProvider().GetMetric(aui.AUI_DOCKART_GRADIENT_TYPE) == aui.AUI_GRADIENT_HORIZONTAL)

        elif eid == ID_AllowFloating:
            event.Check((flags & aui.AUI_MGR_ALLOW_FLOATING) != 0)

        elif eid == ID_TransparentDrag:
            event.Check((flags & aui.AUI_MGR_TRANSPARENT_DRAG) != 0)

        elif eid == ID_TransparentHint:
            event.Check((flags & aui.AUI_MGR_TRANSPARENT_HINT) != 0)

        elif eid == ID_VenetianBlindsHint:
            event.Check((flags & aui.AUI_MGR_VENETIAN_BLINDS_HINT) != 0)

        elif eid == ID_RectangleHint:
            event.Check((flags & aui.AUI_MGR_RECTANGLE_HINT) != 0)

        elif eid == ID_NoHint:
            event.Check(((aui.AUI_MGR_TRANSPARENT_HINT |
                          aui.AUI_MGR_VENETIAN_BLINDS_HINT |
                          aui.AUI_MGR_RECTANGLE_HINT) & flags) == 0)

        elif eid == ID_HintFade:
            event.Check((flags & aui.AUI_MGR_HINT_FADE) != 0);

        elif eid == ID_NoVenetianFade:
            event.Check((flags & aui.AUI_MGR_NO_VENETIAN_BLINDS_FADE) != 0);




    def OnCreatePerspective(self, event):

        dlg = wx.TextEntryDialog(self, "Enter a name for the new perspective:", "AUI Test")

        dlg.SetValue(("Perspective %d")%(len(self._perspectives)+1))
        if dlg.ShowModal() != wx.ID_OK:
            return

        if len(self._perspectives) == 0:
            self._perspectives_menu.AppendSeparator()

        self._perspectives_menu.Append(ID_FirstPerspective[len(self._perspectives)], dlg.GetValue())
        self._perspectives.append(self._mgr.SavePerspective())


    def OnCopyPerspective(self, event):

        s = self._mgr.SavePerspective()

        if wx.TheClipboard.Open():

            wx.TheClipboard.SetData(wx.TextDataObject(s))
            wx.TheClipboard.Close()

    def OnRestorePerspective(self, event):

        self._mgr.LoadPerspective(self._perspectives[event.GetId() - ID_FirstPerspective[0].Value])


    def GetStartPosition(self):

        self.x = self.x + 20
        x = self.x
        pt = self.ClientToScreen(wx.Point(0, 0))

        return wx.Point(pt.x + x, pt.y + x)


    def OnCreateTree(self, event):
        self._mgr.AddPane(self.CreateTreeCtrl(), aui.AuiPaneInfo().
                          Caption("Tree Control").
                          Float().FloatingPosition(self.GetStartPosition()).
                          FloatingSize(wx.Size(150, 300)).CloseButton(True).MaximizeButton(True))
        self._mgr.Update()


    def OnCreateGrid(self, event):
        self._mgr.AddPane(self.CreateGrid(), aui.AuiPaneInfo().
                          Caption("Grid").
                          Float().FloatingPosition(self.GetStartPosition()).
                          FloatingSize(wx.Size(300, 200)).CloseButton(True).MaximizeButton(True))
        self._mgr.Update()


    def OnCreateHTML(self, event):
        self._mgr.AddPane(self.CreateHTMLCtrl(), aui.AuiPaneInfo().
                          Caption("HTML Content").
                          Float().FloatingPosition(self.GetStartPosition()).
                          FloatingSize(wx.Size(300, 200)).CloseButton(True).MaximizeButton(True))
        self._mgr.Update()


    def OnCreateText(self, event):
        self._mgr.AddPane(self.CreateTextCtrl(), aui.AuiPaneInfo().
                          Caption("Text Control").
                          Float().FloatingPosition(self.GetStartPosition()).
                          CloseButton(True).MaximizeButton(True))
        self._mgr.Update()


    def OnNavData(self, event):
        self._mgr.AddPane(self.CreateTextCtrl(), aui.AuiPaneInfo().
                          Caption("Text Control").
                          Float().FloatingPosition(self.GetStartPosition()).
                          CloseButton(True).MaximizeButton(True))

        self._mgr.GetPane("grid_content").Show(event.GetId() == ID_GridContent)
        self._mgr.GetPane("text_content").Show(event.GetId() == ID_TextContent)
        self._mgr.GetPane("tree_content").Show(event.GetId() == ID_TreeContent)
        self._mgr.GetPane("sizereport_content").Show(event.GetId() == ID_SizeReportContent)
        self._mgr.GetPane("html_content").Show(event.GetId() == ID_HTMLContent)
        self._mgr.Update()

    def OnNavModel(self, event):
        self._mgr.AddPane(self.CreateTextCtrl(), aui.AuiPaneInfo().
                          Caption("Text Control").
                          Float().FloatingPosition(self.GetStartPosition()).
                          CloseButton(True).MaximizeButton(True))
        self._mgr.Update()

    def OnNavExpr(self, event):
        self._mgr.AddPane(self.CreateTextCtrl(), aui.AuiPaneInfo().
                          Caption("Text Control").
                          Float().FloatingPosition(self.GetStartPosition()).
                          CloseButton(True).MaximizeButton(True))
        self._mgr.Update()

    def OnNavResults(self, event):
        self._mgr.AddPane(self.CreateTextCtrl(), aui.AuiPaneInfo().
                          Caption("Text Control").
                          Float().FloatingPosition(self.GetStartPosition()).
                          CloseButton(True).MaximizeButton(True))
        self._mgr.Update()

    def OnCreateSizeReport(self, event):
        self._mgr.AddPane(self.CreateSizeReportCtrl(), aui.AuiPaneInfo().
                          Caption("Client Size Reporter").
                          Float().FloatingPosition(self.GetStartPosition()).
                          CloseButton(True).MaximizeButton(True))
        self._mgr.Update()


    def OnChangeContentPane(self, event):

        self._mgr.GetPane("grid_content").Show(event.GetId() == ID_GridContent)
        self._mgr.GetPane("text_content").Show(event.GetId() == ID_TextContent)
        self._mgr.GetPane("tree_content").Show(event.GetId() == ID_TreeContent)
        self._mgr.GetPane("sizereport_content").Show(event.GetId() == ID_SizeReportContent)
        self._mgr.GetPane("html_content").Show(event.GetId() == ID_HTMLContent)
        self._mgr.Update()


    def CreateTextCtrl(self):

        text = ("This is text box %d")%(self.n + 1)

        return wx.TextCtrl(self,-1, text, wx.Point(0, 0), wx.Size(150, 90),
                           wx.NO_BORDER | wx.TE_MULTILINE)

    def CreateStatusTextCtrl(self):

        text = ("Welcome to Message Gateway Simulator!")

        return wx.TextCtrl(self,-1, text, wx.Point(0, 0), wx.Size(150, 90),
                           wx.TE_READONLY | wx.NO_BORDER | wx.TE_MULTILINE)



    def CreateGrid(self):

        grid = wx.grid.Grid(self, -1, wx.Point(0, 0), wx.Size(150, 250),
                            wx.NO_BORDER | wx.WANTS_CHARS)

        grid.CreateGrid(50, 20)

        return grid

    def CreateTreeCtrl(self):

        tree = wx.TreeCtrl(self, -1, wx.Point(0, 0), wx.Size(160, 250),
                           wx.TR_DEFAULT_STYLE | wx.NO_BORDER)

        root = tree.AddRoot("AUI Project")
        items = []

        imglist = wx.ImageList(16, 16, True, 2)
        imglist.Add(wx.ArtProvider.GetBitmap(wx.ART_FOLDER, wx.ART_OTHER, wx.Size(16,16)))
        imglist.Add(wx.ArtProvider.GetBitmap(wx.ART_NORMAL_FILE, wx.ART_OTHER, wx.Size(16,16)))
        tree.AssignImageList(imglist)

        items.append(tree.AppendItem(root, "Item 1", 0))
        items.append(tree.AppendItem(root, "Item 2", 0))
        items.append(tree.AppendItem(root, "Item 3", 0))
        items.append(tree.AppendItem(root, "Item 4", 0))
        items.append(tree.AppendItem(root, "Item 5", 0))

        for ii in range(len(items)):

            id = items[ii]
            tree.AppendItem(id, "Subitem 1", 1)
            tree.AppendItem(id, "Subitem 2", 1)
            tree.AppendItem(id, "Subitem 3", 1)
            tree.AppendItem(id, "Subitem 4", 1)
            tree.AppendItem(id, "Subitem 5", 1)

        tree.Expand(root)

        return tree


    def CreateDataTreeCtrl(self):

        tree = wx.TreeCtrl(self, -1, wx.Point(0, 0), wx.Size(160, 250),
                           wx.TR_DEFAULT_STYLE | wx.NO_BORDER)

        root = tree.AddRoot("Data Tree Control")
        items = []

        imglist = wx.ImageList(16, 16, True, 2)
        imglist.Add(wx.ArtProvider.GetBitmap(wx.ART_FOLDER, wx.ART_OTHER, wx.Size(16,16)))
        imglist.Add(wx.ArtProvider.GetBitmap(wx.ART_NORMAL_FILE, wx.ART_OTHER, wx.Size(16,16)))
        tree.AssignImageList(imglist)

        items.append(tree.AppendItem(root, "Item 1", 0))
        items.append(tree.AppendItem(root, "Item 2", 0))
        items.append(tree.AppendItem(root, "Item 3", 0))
        items.append(tree.AppendItem(root, "Item 4", 0))
        items.append(tree.AppendItem(root, "Item 5", 0))

        for ii in range(len(items)):

            id = items[ii]
            tree.AppendItem(id, "Subitem 1", 1)
            tree.AppendItem(id, "Subitem 2", 1)
            tree.AppendItem(id, "Subitem 3", 1)
            tree.AppendItem(id, "Subitem 4", 1)
            tree.AppendItem(id, "Subitem 5", 1)

        tree.Expand(root)

        return tree


    def CreateSizeReportCtrl(self, width=80, height=80):

        ctrl = SizeReportCtrl(self, -1, wx.DefaultPosition,
                              wx.Size(width, height), self._mgr)
        return ctrl


    def CreateHTMLCtrl(self):
        ctrl = wx.html.HtmlWindow(self, -1, wx.DefaultPosition, wx.Size(400, 300))
        if "gtk2" in wx.PlatformInfo or "gtk3" in wx.PlatformInfo:
            ctrl.SetStandardFonts()
        ctrl.SetPage(self.GetIntroText())
        return ctrl


    def GetIntroText(self):
        return overview


# -- wx.SizeReportCtrl --
# (a utility control that always reports it's client size)

class SizeReportCtrl(wx.Control):

    def __init__(self, parent, id=wx.ID_ANY, pos=wx.DefaultPosition,
                 size=wx.DefaultSize, mgr=None):

        wx.Control.__init__(self, parent, id, pos, size, wx.NO_BORDER)

        self._mgr = mgr

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)


    def OnPaint(self, event):

        dc = wx.PaintDC(self)

        size = self.GetClientSize()
        s = ("Size: %d x %d")%(size.x, size.y)

        dc.SetFont(wx.NORMAL_FONT)
        w, height = dc.GetTextExtent(s)
        height = height + 3
        dc.SetBrush(wx.WHITE_BRUSH)
        dc.SetPen(wx.WHITE_PEN)
        dc.DrawRectangle(0, 0, size.x, size.y)
        dc.SetPen(wx.LIGHT_GREY_PEN)
        dc.DrawLine(0, 0, size.x, size.y)
        dc.DrawLine(0, size.y, size.x, 0)
        dc.DrawText(s, (size.x-w)/2, ((size.y-(height*5))/2))

        if self._mgr:

            pi = self._mgr.GetPane(self)

            s = ("Layer: %d")%pi.dock_layer
            w, h = dc.GetTextExtent(s)
            dc.DrawText(s, (size.x-w)/2, ((size.y-(height*5))/2)+(height*1))

            s = ("Dock: %d Row: %d")%(pi.dock_direction, pi.dock_row)
            w, h = dc.GetTextExtent(s)
            dc.DrawText(s, (size.x-w)/2, ((size.y-(height*5))/2)+(height*2))

            s = ("Position: %d")%pi.dock_pos
            w, h = dc.GetTextExtent(s)
            dc.DrawText(s, (size.x-w)/2, ((size.y-(height*5))/2)+(height*3))

            s = ("Proportion: %d")%pi.dock_proportion
            w, h = dc.GetTextExtent(s)
            dc.DrawText(s, (size.x-w)/2, ((size.y-(height*5))/2)+(height*4))


    def OnEraseBackground(self, event):
        # intentionally empty
        pass


    def OnSize(self, event):

        self.Refresh()
        event.Skip()


ID_PaneBorderSize = wx.ID_HIGHEST + 1
ID_SashSize = ID_PaneBorderSize + 1
ID_CaptionSize = ID_PaneBorderSize + 2
ID_BackgroundColor = ID_PaneBorderSize + 3
ID_SashColor = ID_PaneBorderSize + 4
ID_InactiveCaptionColor =  ID_PaneBorderSize + 5
ID_InactiveCaptionGradientColor = ID_PaneBorderSize + 6
ID_InactiveCaptionTextColor = ID_PaneBorderSize + 7
ID_ActiveCaptionColor = ID_PaneBorderSize + 8
ID_ActiveCaptionGradientColor = ID_PaneBorderSize + 9
ID_ActiveCaptionTextColor = ID_PaneBorderSize + 10
ID_BorderColor = ID_PaneBorderSize + 11
ID_GripperColor = ID_PaneBorderSize + 12

class SettingsPanel(wx.Panel):

    def __init__(self, parent, frame):

        wx.Panel.__init__(self, parent, wx.ID_ANY, wx.DefaultPosition,
                          wx.DefaultSize)

        self._frame = frame

        vert = wx.BoxSizer(wx.VERTICAL)

        s1 = wx.BoxSizer(wx.HORIZONTAL)
        self._border_size = wx.SpinCtrl(self, ID_PaneBorderSize, "", wx.DefaultPosition, wx.Size(50,20))
        s1.Add((1, 1), 1, wx.EXPAND)
        s1.Add(wx.StaticText(self, -1, "Pane Border Size:"))
        s1.Add(self._border_size)
        s1.Add((1, 1), 1, wx.EXPAND)
        s1.SetItemMinSize(1, (180, 20))
        #vert.Add(s1, 0, wx.EXPAND | wxLEFT | wxBOTTOM, 5)

        s2 = wx.BoxSizer(wx.HORIZONTAL)
        self._sash_size = wx.SpinCtrl(self, ID_SashSize, "", wx.DefaultPosition, wx.Size(50,20))
        s2.Add((1, 1), 1, wx.EXPAND)
        s2.Add(wx.StaticText(self, -1, "Sash Size:"))
        s2.Add(self._sash_size)
        s2.Add((1, 1), 1, wx.EXPAND)
        s2.SetItemMinSize(1, (180, 20))
        #vert.Add(s2, 0, wx.EXPAND | wxLEFT | wxBOTTOM, 5)

        s3 = wx.BoxSizer(wx.HORIZONTAL)
        self._caption_size = wx.SpinCtrl(self, ID_CaptionSize, "", wx.DefaultPosition, wx.Size(50,20))
        s3.Add((1, 1), 1, wx.EXPAND)
        s3.Add(wx.StaticText(self, -1, "Caption Size:"))
        s3.Add(self._caption_size)
        s3.Add((1, 1), 1, wx.EXPAND)
        s3.SetItemMinSize(1, (180, 20))
        #vert.Add(s3, 0, wx.EXPAND | wxLEFT | wxBOTTOM, 5)

        #vert.Add(1, 1, 1, wx.EXPAND)

        b = self.CreateColorBitmap(wx.BLACK)

        s4 = wx.BoxSizer(wx.HORIZONTAL)
        self._background_color = wx.BitmapButton(self, ID_BackgroundColor, b, wx.DefaultPosition, wx.Size(50,25))
        s4.Add((1, 1), 1, wx.EXPAND)
        s4.Add(wx.StaticText(self, -1, "Background Color:"))
        s4.Add(self._background_color)
        s4.Add((1, 1), 1, wx.EXPAND)
        s4.SetItemMinSize(1, (180, 20))

        s5 = wx.BoxSizer(wx.HORIZONTAL)
        self._sash_color = wx.BitmapButton(self, ID_SashColor, b, wx.DefaultPosition, wx.Size(50,25))
        s5.Add((1, 1), 1, wx.EXPAND)
        s5.Add(wx.StaticText(self, -1, "Sash Color:"))
        s5.Add(self._sash_color)
        s5.Add((1, 1), 1, wx.EXPAND)
        s5.SetItemMinSize(1, (180, 20))

        s6 = wx.BoxSizer(wx.HORIZONTAL)
        self._inactive_caption_color = wx.BitmapButton(self, ID_InactiveCaptionColor, b,
                                                       wx.DefaultPosition, wx.Size(50,25))
        s6.Add((1, 1), 1, wx.EXPAND)
        s6.Add(wx.StaticText(self, -1, "Normal Caption:"))
        s6.Add(self._inactive_caption_color)
        s6.Add((1, 1), 1, wx.EXPAND)
        s6.SetItemMinSize(1, (180, 20))

        s7 = wx.BoxSizer(wx.HORIZONTAL)
        self._inactive_caption_gradient_color = wx.BitmapButton(self, ID_InactiveCaptionGradientColor,
                                                                b, wx.DefaultPosition, wx.Size(50,25))
        s7.Add((1, 1), 1, wx.EXPAND)
        s7.Add(wx.StaticText(self, -1, "Normal Caption Gradient:"))
        s7.Add(self._inactive_caption_gradient_color)
        s7.Add((1, 1), 1, wx.EXPAND)
        s7.SetItemMinSize(1, (180, 20))

        s8 = wx.BoxSizer(wx.HORIZONTAL)
        self._inactive_caption_text_color = wx.BitmapButton(self, ID_InactiveCaptionTextColor, b,
                                                            wx.DefaultPosition, wx.Size(50,25))
        s8.Add((1, 1), 1, wx.EXPAND)
        s8.Add(wx.StaticText(self, -1, "Normal Caption Text:"))
        s8.Add(self._inactive_caption_text_color)
        s8.Add((1, 1), 1, wx.EXPAND)
        s8.SetItemMinSize(1, (180, 20))

        s9 = wx.BoxSizer(wx.HORIZONTAL)
        self._active_caption_color = wx.BitmapButton(self, ID_ActiveCaptionColor, b,
                                                     wx.DefaultPosition, wx.Size(50,25))
        s9.Add((1, 1), 1, wx.EXPAND)
        s9.Add(wx.StaticText(self, -1, "Active Caption:"))
        s9.Add(self._active_caption_color)
        s9.Add((1, 1), 1, wx.EXPAND)
        s9.SetItemMinSize(1, (180, 20))

        s10 = wx.BoxSizer(wx.HORIZONTAL)
        self._active_caption_gradient_color = wx.BitmapButton(self, ID_ActiveCaptionGradientColor,
                                                              b, wx.DefaultPosition, wx.Size(50,25))
        s10.Add((1, 1), 1, wx.EXPAND)
        s10.Add(wx.StaticText(self, -1, "Active Caption Gradient:"))
        s10.Add(self._active_caption_gradient_color)
        s10.Add((1, 1), 1, wx.EXPAND)
        s10.SetItemMinSize(1, (180, 20))

        s11 = wx.BoxSizer(wx.HORIZONTAL)
        self._active_caption_text_color = wx.BitmapButton(self, ID_ActiveCaptionTextColor,
                                                          b, wx.DefaultPosition, wx.Size(50,25))
        s11.Add((1, 1), 1, wx.EXPAND)
        s11.Add(wx.StaticText(self, -1, "Active Caption Text:"))
        s11.Add(self._active_caption_text_color)
        s11.Add((1, 1), 1, wx.EXPAND)
        s11.SetItemMinSize(1, (180, 20))

        s12 = wx.BoxSizer(wx.HORIZONTAL)
        self._border_color = wx.BitmapButton(self, ID_BorderColor, b, wx.DefaultPosition,
                                             wx.Size(50,25))
        s12.Add((1, 1), 1, wx.EXPAND)
        s12.Add(wx.StaticText(self, -1, "Border Color:"))
        s12.Add(self._border_color)
        s12.Add((1, 1), 1, wx.EXPAND)
        s12.SetItemMinSize(1, (180, 20))

        s13 = wx.BoxSizer(wx.HORIZONTAL)
        self._gripper_color = wx.BitmapButton(self, ID_GripperColor, b, wx.DefaultPosition,
                                              wx.Size(50,25))
        s13.Add((1, 1), 1, wx.EXPAND)
        s13.Add(wx.StaticText(self, -1, "Gripper Color:"))
        s13.Add(self._gripper_color)
        s13.Add((1, 1), 1, wx.EXPAND)
        s13.SetItemMinSize(1, (180, 20))

        grid_sizer = wx.GridSizer(cols=2)
        grid_sizer.SetHGap(5)
        grid_sizer.Add(s1)
        grid_sizer.Add(s4)
        grid_sizer.Add(s2)
        grid_sizer.Add(s5)
        grid_sizer.Add(s3)
        grid_sizer.Add(s13)
        grid_sizer.Add((1, 1))
        grid_sizer.Add(s12)
        grid_sizer.Add(s6)
        grid_sizer.Add(s9)
        grid_sizer.Add(s7)
        grid_sizer.Add(s10)
        grid_sizer.Add(s8)
        grid_sizer.Add(s11)

        cont_sizer = wx.BoxSizer(wx.VERTICAL)
        cont_sizer.Add(grid_sizer, 1, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(cont_sizer)
        self.GetSizer().SetSizeHints(self)

        self._border_size.SetValue(frame.GetDockArt().GetMetric(aui.AUI_DOCKART_PANE_BORDER_SIZE))
        self._sash_size.SetValue(frame.GetDockArt().GetMetric(aui.AUI_DOCKART_SASH_SIZE))
        self._caption_size.SetValue(frame.GetDockArt().GetMetric(aui.AUI_DOCKART_CAPTION_SIZE))

        self.UpdateColors()

        self.Bind(wx.EVT_SPINCTRL, self.OnPaneBorderSize, id=ID_PaneBorderSize)
        self.Bind(wx.EVT_SPINCTRL, self.OnSashSize, id=ID_SashSize)
        self.Bind(wx.EVT_SPINCTRL, self.OnCaptionSize, id=ID_CaptionSize)
        self.Bind(wx.EVT_BUTTON, self.OnSetColor, id=ID_BackgroundColor)
        self.Bind(wx.EVT_BUTTON, self.OnSetColor, id=ID_SashColor)
        self.Bind(wx.EVT_BUTTON, self.OnSetColor, id=ID_InactiveCaptionColor)
        self.Bind(wx.EVT_BUTTON, self.OnSetColor, id=ID_InactiveCaptionGradientColor)
        self.Bind(wx.EVT_BUTTON, self.OnSetColor, id=ID_InactiveCaptionTextColor)
        self.Bind(wx.EVT_BUTTON, self.OnSetColor, id=ID_ActiveCaptionColor)
        self.Bind(wx.EVT_BUTTON, self.OnSetColor, id=ID_ActiveCaptionGradientColor)
        self.Bind(wx.EVT_BUTTON, self.OnSetColor, id=ID_ActiveCaptionTextColor)
        self.Bind(wx.EVT_BUTTON, self.OnSetColor, id=ID_BorderColor)
        self.Bind(wx.EVT_BUTTON, self.OnSetColor, id=ID_GripperColor)


    def CreateColorBitmap(self, c):
        image = wx.Image(25, 14)

        for x in range(25):
            for y in range(14):
                pixcol = c
                if x == 0 or x == 24 or y == 0 or y == 13:
                    pixcol = wx.BLACK

                image.SetRGB(x, y, pixcol.Red(), pixcol.Green(), pixcol.Blue())

        return image.ConvertToBitmap()


    def UpdateColors(self):

        bk = self._frame.GetDockArt().GetColour(aui.AUI_DOCKART_BACKGROUND_COLOUR)
        self._background_color.SetBitmapLabel(self.CreateColorBitmap(bk))

        cap = self._frame.GetDockArt().GetColour(aui.AUI_DOCKART_INACTIVE_CAPTION_COLOUR)
        self._inactive_caption_color.SetBitmapLabel(self.CreateColorBitmap(cap))

        capgrad = self._frame.GetDockArt().GetColour(aui.AUI_DOCKART_INACTIVE_CAPTION_GRADIENT_COLOUR)
        self._inactive_caption_gradient_color.SetBitmapLabel(self.CreateColorBitmap(capgrad))

        captxt = self._frame.GetDockArt().GetColour(aui.AUI_DOCKART_INACTIVE_CAPTION_TEXT_COLOUR)
        self._inactive_caption_text_color.SetBitmapLabel(self.CreateColorBitmap(captxt))

        acap = self._frame.GetDockArt().GetColour(aui.AUI_DOCKART_ACTIVE_CAPTION_COLOUR)
        self._active_caption_color.SetBitmapLabel(self.CreateColorBitmap(acap))

        acapgrad = self._frame.GetDockArt().GetColour(aui.AUI_DOCKART_ACTIVE_CAPTION_GRADIENT_COLOUR)
        self._active_caption_gradient_color.SetBitmapLabel(self.CreateColorBitmap(acapgrad))

        acaptxt = self._frame.GetDockArt().GetColour(aui.AUI_DOCKART_ACTIVE_CAPTION_TEXT_COLOUR)
        self._active_caption_text_color.SetBitmapLabel(self.CreateColorBitmap(acaptxt))

        sash = self._frame.GetDockArt().GetColour(aui.AUI_DOCKART_SASH_COLOUR)
        self._sash_color.SetBitmapLabel(self.CreateColorBitmap(sash))

        border = self._frame.GetDockArt().GetColour(aui.AUI_DOCKART_BORDER_COLOUR)
        self._border_color.SetBitmapLabel(self.CreateColorBitmap(border))

        gripper = self._frame.GetDockArt().GetColour(aui.AUI_DOCKART_GRIPPER_COLOUR)
        self._gripper_color.SetBitmapLabel(self.CreateColorBitmap(gripper))


    def OnPaneBorderSize(self, event):

        self._frame.GetDockArt().SetMetric(aui.AUI_DOCKART_PANE_BORDER_SIZE,
                                           event.GetInt())
        self._frame.DoUpdate()


    def OnSashSize(self, event):

        self._frame.GetDockArt().SetMetric(aui.AUI_DOCKART_SASH_SIZE,
                                           event.GetInt())
        self._frame.DoUpdate()


    def OnCaptionSize(self, event):

        self._frame.GetDockArt().SetMetric(aui.AUI_DOCKART_CAPTION_SIZE,
                                           event.GetInt())
        self._frame.DoUpdate()


    def OnSetColor(self, event):

        dlg = wx.ColourDialog(self._frame)

        dlg.SetTitle("Color Picker")

        if dlg.ShowModal() != wx.ID_OK:
            return

        var = 0
        if event.GetId() == ID_BackgroundColor:
            var = aui.AUI_DOCKART_BACKGROUND_COLOUR
        elif event.GetId() == ID_SashColor:
            var = aui.AUI_DOCKART_SASH_COLOUR
        elif event.GetId() == ID_InactiveCaptionColor:
            var = aui.AUI_DOCKART_INACTIVE_CAPTION_COLOUR
        elif event.GetId() == ID_InactiveCaptionGradientColor:
            var = aui.AUI_DOCKART_INACTIVE_CAPTION_GRADIENT_COLOUR
        elif event.GetId() == ID_InactiveCaptionTextColor:
            var = aui.AUI_DOCKART_INACTIVE_CAPTION_TEXT_COLOUR
        elif event.GetId() == ID_ActiveCaptionColor:
            var = aui.AUI_DOCKART_ACTIVE_CAPTION_COLOUR
        elif event.GetId() == ID_ActiveCaptionGradientColor:
            var = aui.AUI_DOCKART_ACTIVE_CAPTION_GRADIENT_COLOUR
        elif event.GetId() == ID_ActiveCaptionTextColor:
            var = aui.AUI_DOCKART_ACTIVE_CAPTION_TEXT_COLOUR
        elif event.GetId() == ID_BorderColor:
            var = aui.AUI_DOCKART_BORDER_COLOUR
        elif event.GetId() == ID_GripperColor:
            var = aui.AUI_DOCKART_GRIPPER_COLOUR
        else:
            return

        self._frame.GetDockArt().SetColor(var, dlg.GetColourData().GetColour())
        self._frame.DoUpdate()
        self.UpdateColors()


#----------------------------------------------------------------------



overview = """\
<html><body>
<h3>Munvo MGSim, the Message Gateway Simulator module</h3>

<br/><b>Overview</b><br/>

<p>This program is a Message Gateway Simulator which produces referential models
for iterative training.  This simulator provides a workbench to tweak models
to meet preferred goals.  The generates patterns that can be used to deploy
on Message Gateway to validate accuracy of results.</p>

<p><b>Features</b></p>

<ul>
<li>Native, dockable floating frames</li>
<li>wxPython to manage cross-platform window management</li>
<li>PostgreSQL server to manage internal system metadata</li>
<li>TensorFlow for deep reinforcement learning</li>
<li>wx.OGL for 2D graphic management</li>
<li>GLUT/OpenGL for any optional 3d rendering (model visualizaton)</li>
<li>C++ for Message Gateway Simulator core code</li>
</ul>

</body></html>
"""

#----------------------------------------------------------------------




class MainLayout(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)

        self.SetAutoLayout(True)
        self.Bind(wx.EVT_BUTTON, self.OnButton)

        self.panelA = wx.Window(self, -1, style=wx.SIMPLE_BORDER)
        self.panelA.SetBackgroundColour(wx.BLUE)
        self.panelA.SetConstraints(
            layoutf.Layoutf('t=t10#1;l=l10#1;b=b10#1;r%r50#1',(self,))
        )

        self.panelB = wx.Window(self, -1, style=wx.SIMPLE_BORDER)
        self.panelB.SetBackgroundColour(wx.RED)
        self.panelB.SetConstraints(
            layoutf.Layoutf('t=t10#1;r=r10#1;b%b30#1;l>10#2', (self,self.panelA))
        )

        self.panelC = wx.Window(self, -1, style=wx.SIMPLE_BORDER)
        self.panelC.SetBackgroundColour(wx.WHITE)
        self.panelC.SetConstraints(
            layoutf.Layoutf('t_10#3;r=r10#1;b=b10#1;l>10#2', (self,self.panelA,self.panelB))
        )

        b = wx.Button(self.panelA, -1, 'Train')
        b.SetConstraints(layoutf.Layoutf('t=t2#1;l=l2#1;h*;w%w50#1', (self.panelA,)))

        b = wx.Button(self.panelA, -1, 'Run')
        b.SetConstraints(layoutf.Layoutf('t=t2#1;l=l2#1;h*;w%w50#1', (self.panelA,)))

        b = wx.Button(self.panelA, -1, 'Stop')
        b.SetConstraints(layoutf.Layoutf('t=t2#1;l=l2#1;h*;w%w50#1', (self.panelA,)))

        b = wx.Button(self.panelB, -1, ' Panel B ')
        b.SetConstraints(layoutf.Layoutf('t=t2#1;r=r4#1;h*;w*', (self.panelB,)))

        self.panelD = wx.Window(self.panelC, -1, style=wx.SIMPLE_BORDER)
        self.panelD.SetBackgroundColour(wx.GREEN)
        self.panelD.SetConstraints(
            layoutf.Layoutf('b%h50#1;r%w50#1;h=h#2;w=w#2', (self.panelC, b))
        )

        b = wx.Button(self.panelC, -1, ' Panel C ')
        b.SetConstraints(layoutf.Layoutf('t_#1;l>#1;h*;w*', (self.panelD,)))

        wx.StaticText(self.panelD, -1, "Panel D", (4, 4)).SetBackgroundColour(wx.GREEN)

    def OnButton(self, event):
        wx.Bell()

#---------------------------------------------------------------------------
class MainPanel(wx.Panel):
    def __init__(self, parent, log):
        self.log = log
        wx.Panel.__init__(self, parent, -1)
#        b = wx.Button(self, -1, "Show the aui Demo Frame", (50,50))
#        self.Bind(wx.EVT_BUTTON, self.OnButton, b)
        frame = PyAUIFrame(self, wx.ID_ANY, "Message Gateway Simulator", size=(1024, 768))
        frame.Show()

class RunWxApp(wx.App):
    def __init__(self, name):
        self.name = name
        self.log = Log()
        wx.App.__init__(self, redirect=False)


    def OnInit(self):
        wx.Log.SetActiveTarget(wx.LogStderr())

        self.SetAssertMode(assertMode)

        baseFrame = wx.Frame(None, -1, self.name, size=(0,0),
                         style=wx.DEFAULT_FRAME_STYLE, name=self.name)
        baseFrame.Bind(wx.EVT_CLOSE, self.OnCloseFrame)
        win = MainLayout(MainPanel(baseFrame, self.log))

        win.SetFocus()
        self.window = win
        self.SetTopWindow(baseFrame)
        self.frame = baseFrame
        #wx.Log.SetActiveTarget(wx.LogStderr())
        #wx.Log.SetTraceMask(wx.TraceMessages)
        return True


    def OnExitApp(self, evt):
        self.frame.Close(True)


    def OnCloseFrame(self, evt):
        # Shutdown main loop
        Logger.print('OnCloseFrame: %s' % evt)
#        if hasattr(self, "window") and hasattr(self.window, "ShutdownDemo"):
#            self.window.ShutdownDemo()
        evt.Skip()

    def OnLoadModel(self, evt):
        a = 1
        #wx.lib.inspection.InspectionTool().Show()

def build_arg_parser(args):
    arg_parser = ArgParser()
    Logger.print('args: %s' % args)
    arg_parser.load_args(args)

    arg_file = arg_parser.parse_string('arg_file', '')
    Logger.print('arg_file: %s' % arg_file)
    if (arg_file != ''):
        succ = arg_parser.load_file(arg_file)
        assert succ, Logger.print('Failed to load args from: ' + arg_file)

    rand_seed_key = 'rand_seed'
    if (arg_parser.has_key(rand_seed_key)):
        rand_seed = arg_parser.parse_int(rand_seed_key)
        rand_seed += 1000 * MPIUtil.get_proc_rank()
        Util.set_global_seeds(rand_seed)

    return arg_parser

def update_intermediate_buffer():
    if not (reshaping):
        if (win_width != world.env.get_win_width() or win_height != world.env.get_win_height()):
            world.env.reshape(win_width, win_height)

    return

def update_world(world, time_elapsed):
    num_substeps = world.env.get_num_update_substeps()
    timestep = time_elapsed / num_substeps
    num_substeps = 1 if (time_elapsed == 0) else num_substeps

    for i in range(num_substeps):
        world.update(timestep)

        valid_episode = world.env.check_valid_episode()
        if valid_episode:
            end_episode = world.env.is_episode_end()
            if (end_episode):
                Logger.print('[Main] update_world(): End of episode: ' + str(timestep))
                world.end_episode()
                world.reset()
                break
        else:
            world.reset()
            break
    return

def draw():
    global reshaping

    update_intermediate_buffer()
    world.env.draw()
    
    glutSwapBuffers()
    reshaping = False

    return

def draw2():
    global reshaping

    #update_intermediate_buffer()
    #world.env.draw()

    #glutSwapBuffers()
    reshaping = False

    return


def reshape(w, h):
    global reshaping
    global win_width
    global win_height

    reshaping = True
    win_width = w
    win_height = h

    return

def step_anim(timestep):
    global animating
    global world

    update_world(world, timestep)
    animating = False
    glutPostRedisplay()
    return

def reload():
    global world
    global args

    world = build_world(args)
    return

def reset():
    world.reset()
    return

def get_num_timesteps():
    global playback_speed

    num_steps = int(playback_speed)
    if (num_steps == 0):
        num_steps = 1

    num_steps = np.abs(num_steps)
    return num_steps

def calc_display_anim_time(num_timestes):
    global display_anim_time
    global playback_speed

    anim_time = int(display_anim_time * num_timestes / playback_speed)
    anim_time = np.abs(anim_time)
    return anim_time

def shutdown():
    global world

    Logger.print('Shutting down...')
    world.shutdown()
    sys.exit(0)
    return

def get_curr_time():
    curr_time = glutGet(GLUT_ELAPSED_TIME)
    return curr_time

def init_time():
    global prev_time
    global updates_per_sec
    prev_time = get_curr_time()
    updates_per_sec = 0
    return

def animate(callback_val):
    global prev_time
    global updates_per_sec
    global world

    counter_decay = 0

    if (animating):
        num_steps = get_num_timesteps()
        curr_time = get_curr_time()
        time_elapsed = curr_time - prev_time
        prev_time = curr_time;

        timestep = -update_timestep if (playback_speed < 0) else update_timestep
        for i in range(num_steps):
            update_world(world, timestep)
        
        # FPS counting
        update_count = num_steps / (0.001 * time_elapsed)
        if (np.isfinite(update_count)):
            updates_per_sec = counter_decay * updates_per_sec + (1 - counter_decay) * update_count;
            world.env.set_updates_per_sec(updates_per_sec);
            
        timer_step = calc_display_anim_time(num_steps)
        update_dur = get_curr_time() - curr_time
        timer_step -= update_dur
        timer_step = np.maximum(timer_step, 0)
        
        glutTimerFunc(int(timer_step), animate, 0)
        glutPostRedisplay()

    if (world.env.is_done()):
        shutdown()

    return

def toggle_animate():
    global animating

    animating = not animating
    if (animating):
        glutTimerFunc(display_anim_time, animate, 0)

    return

def change_playback_speed(delta):
    global playback_speed

    prev_playback = playback_speed
    playback_speed += delta
    world.env.set_playback_speed(playback_speed)

    if (np.abs(prev_playback) < 0.0001 and np.abs(playback_speed) > 0.0001):
        glutTimerFunc(display_anim_time, animate, 0)

    return

def toggle_training():
    global world

    world.enable_training = not world.enable_training
    if (world.enable_training):
        Logger.print('Training enabled')
    else:
        Logger.print('Training disabled')
    return

def keyboard(key, x, y):
    key_val = int.from_bytes(key, byteorder='big')
    world.env.keyboard(key_val, x, y)

    if (key == b'\x1b'): # escape
        shutdown()
    elif (key == b' '):
        toggle_animate();
    elif (key == b'>'):
        step_anim(update_timestep);
    elif (key == b'<'):
        step_anim(-update_timestep);
    elif (key == b','):
        change_playback_speed(-playback_delta);
    elif (key == b'.'):
        change_playback_speed(playback_delta);
    elif (key == b'/'):
        change_playback_speed(-playback_speed + 1);
    elif (key == b'l'):
        reload();
    elif (key == b'r'):
        reset();
    elif (key == b't'):
        toggle_training()

    glutPostRedisplay()
    return

def mouse_click(button, state, x, y):
    world.env.mouse_click(button, state, x, y)
    glutPostRedisplay()

def mouse_move(x, y):
    world.env.mouse_move(x, y)
    glutPostRedisplay()
    
    return

def init_glut_draw():
    glutInit()  
    
    glutInitContextVersion(3, 2)
    glutInitContextFlags(GLUT_FORWARD_COMPATIBLE)
    glutInitContextProfile(GLUT_CORE_PROFILE)

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(win_width, win_height)
    glutCreateWindow(b'Message Gateway Simulator')
    return
    
def setup_glut_draw():
    glutDisplayFunc(draw)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutMouseFunc(mouse_click)
    glutMotionFunc(mouse_move)
    glutTimerFunc(display_anim_time, animate, 0)

    glutSetOption ( GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION)

#    glClearColor(1,1,1,1)

    mainwin = glutGetWindow()
    winmax=mainwin;

    return

def build_world(args, playback_speed=1):
    arg_parser = build_arg_parser(args)
    Logger.print('[MGSim] Preparing environment.')
    env = MGSimEnv(args, False)
    Logger.print('[MGSim] Preparing RLWorld.')
    world = RLWorld(env, arg_parser)
    world.env.set_playback_speed(playback_speed)
    return world

def draw_main_loop():
    return


def main(argv):
    global args

    '''    if len(argv) < 2:
            print("Please specify a demo module name on the command-line")
            raise SystemExit
    '''
    # Command line arguments
    args = sys.argv[1:]

    name = 'Simulator v0.1'
    app = RunWxApp(name)

#    init_glut_draw()
    reload()
#    setup_glut_draw()
    reshape(win_width, win_height)
    world.env.reshape(win_width, win_height)
    init_time()
#    glutMainLoop()
    app.MainLoop()

    return

if __name__ == "__main__":
    main(sys.argv)
