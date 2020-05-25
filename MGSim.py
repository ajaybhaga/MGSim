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

- wxPython to manage cross-platform window management.
- TensorFlow for deep reinforcement learning
- wx.OGL for 2D graphic management
- GLUT/OpenGL for any optional 3d rendering (model visualizaton)
- C++ for Message Gateway Simulator core code

'''

import numpy as np
import random
import wx
import sys

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


import wx
import wx.lib.inspection
import wx.lib.mixins.inspection
import sys, os

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

# stuff for debugging
print("Python %s" % sys.version)
print("wx.version: %s" % wx.version())
##print("pid: %s" % os.getpid()); input("Press Enter...")

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

        b = wx.Button(self.panelA, -1, ' Panel A ')
        b.SetConstraints(layoutf.Layoutf('X=X#1;Y=Y#1;h*;w%w50#1', (self.panelA,)))

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

class RunWxApp(wx.App):
    def __init__(self, name):
        self.name = name
        self.log = Log()
        wx.App.__init__(self, redirect=False)


    def OnInit(self):
        wx.Log.SetActiveTarget(wx.LogStderr())

        self.SetAssertMode(assertMode)

        frame = wx.Frame(None, -1, "Munvo Message Gateway: " + self.name, size=(200,100),
                         style=wx.DEFAULT_FRAME_STYLE, name="run a sample")
        frame.CreateStatusBar()

        menuBar = wx.MenuBar()
        menu = wx.Menu()
        item = menu.Append(-1, "&Load Model\tF6", "Load model into Simulator")
        self.Bind(wx.EVT_MENU, self.OnLoadModel, item)
        item = menu.Append(wx.ID_EXIT, "E&xit\tCtrl-Q", "Exit demo")
        self.Bind(wx.EVT_MENU, self.OnExitApp, item)
        menuBar.Append(menu, "&File")

        ns = {}
        ns['wx'] = wx
        ns['app'] = self
#        ns['module'] = self.demoModule
        ns['frame'] = frame

        frame.SetMenuBar(menuBar)
        frame.Show(True)
        frame.Bind(wx.EVT_CLOSE, self.OnCloseFrame)

 #       win = self.demoModule.runTest(frame, frame, Log())

        win = MainLayout(frame)

        # a window will be returned if the demo does not create
        # its own top-level window
        if win:
            # so set the frame to a good size for showing stuff
            frame.SetSize((800, 600))
            win.SetFocus()
            self.window = win
            ns['win'] = win
            frect = frame.GetRect()

        else:
            # It was probably a dialog or something that is already
            # gone, so we're done.
            frame.Destroy()
            return True

        self.SetTopWindow(frame)
        self.frame = frame
        #wx.Log.SetActiveTarget(wx.LogStderr())
        #wx.Log.SetTraceMask(wx.TraceMessages)

        return True


    def OnExitApp(self, evt):
        self.frame.Close(True)


    def OnCloseFrame(self, evt):
        if hasattr(self, "window") and hasattr(self.window, "ShutdownDemo"):
            self.window.ShutdownDemo()
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

    world = build_world(args, enable_draw=True)
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

    sw1=glutCreateSubWindow(mainwin,4,240,314,236)
    #glutSpecialFunc(special);
    #glutEntryFunc(entry);
    glutDisplayFunc(draw2)
    glutReshapeFunc(reshape)
#    glutKeyboardFunc(keyboard)
#    glutMouseFunc(mouse_click)
#    glutMotionFunc(mouse_move)

 #   glClearColor(0.7,0.7,0.7,1)

    reshape(win_width, win_height)
    world.env.reshape(win_width, win_height)
    
    return

def build_world(args, enable_draw, playback_speed=1):
    arg_parser = build_arg_parser(args)
    Logger.print('[MGSim] Preparing environment.')
    env = MGSimEnv(args, enable_draw)
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
    init_time()
#    glutMainLoop()

    app.MainLoop()


# ensure the CWD is the demo folder
    #demoFolder = os.path.realpath(os.path.dirname(__file__))
    #os.chdir(demoFolder)

    #sys.path.insert(0, os.path.join(demoFolder, 'agw'))
    #sys.path.insert(0, '.')

    #name, ext  = os.path.splitext(argv[1])


    #app = RunDemoApp(name, module, useShell)
    #app.MainLoop()

    return

buttonDefs = {
    wx.NewIdRef() : ('CubeCanvas', 'Cube'),
    wx.NewIdRef() : ('ConeCanvas', 'Cone'),
}

class ButtonPanel(wx.Panel):
    def __init__(self, parent, log):
        wx.Panel.__init__(self, parent, -1)
        self.log = log

        box = wx.BoxSizer(wx.HORIZONTAL)
        box.Add((20, 30))
        keys = sorted(buttonDefs)
        for k in keys:
            text = buttonDefs[k][1]
            btn = wx.Button(self, k, text)
            box.Add(btn, 0, wx.ALIGN_CENTER|wx.ALL, 15)
            self.Bind(wx.EVT_BUTTON, self.OnButton, btn)

        #** Enable this to show putting a GLCanvas on the wx.Panel
        if 0:
            c = CubeCanvas(self)
            c.SetSize((200, 200))
            box.Add(c, 0, wx.ALIGN_CENTER|wx.ALL, 15)

        self.SetAutoLayout(True)
        self.SetSizer(box)


        if not haveGLCanvas:
            dlg = wx.MessageDialog(self,
                                   'The GLCanvas class has not been included with this build of wxPython!',
                                   'Sorry', wx.OK | wx.ICON_WARNING)
            dlg.ShowModal()
            dlg.Destroy()

        elif not haveOpenGL:
            dlg = wx.MessageDialog(self,
                                   'The OpenGL package was not found.  You can get it at\n'
                                   'http://PyOpenGL.sourceforge.net/',
                                   'Sorry', wx.OK | wx.ICON_WARNING)
            dlg.ShowModal()
            dlg.Destroy()

        else:
            canvasClassName = buttonDefs[evt.GetId()][0]
            canvasClass = eval(canvasClassName)
            frame = wx.Frame(None, -1, canvasClassName, size=(400,400))
            canvas = canvasClass(frame)
            frame.Show(True)


class MyCanvasBase(glcanvas.GLCanvas):
    def __init__(self, parent):
        glcanvas.GLCanvas.__init__(self, parent, -1)
        self.init = False
        self.context = glcanvas.GLContext(self)

        # initial mouse position
        self.lastx = self.x = 30
        self.lasty = self.y = 30
        self.size = None
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnMouseDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnMouseUp)
        self.Bind(wx.EVT_MOTION, self.OnMouseMotion)


    def OnEraseBackground(self, event):
        pass # Do nothing, to avoid flashing on MSW.


    def OnSize(self, event):
        wx.CallAfter(self.DoSetViewport)
        event.Skip()


    def DoSetViewport(self):
        size = self.size = self.GetClientSize() * self.GetContentScaleFactor()
        self.SetCurrent(self.context)
        glViewport(0, 0, size.width, size.height)


    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        self.SetCurrent(self.context)
        if not self.init:
            self.InitGL()
            self.init = True
        self.OnDraw()


    def OnMouseDown(self, evt):
        self.CaptureMouse()
        self.x, self.y = self.lastx, self.lasty = evt.GetPosition()


    def OnMouseUp(self, evt):
        self.ReleaseMouse()


    def OnMouseMotion(self, evt):
        if evt.Dragging() and evt.LeftIsDown():
            self.lastx, self.lasty = self.x, self.y
            self.x, self.y = evt.GetPosition()
            self.Refresh(False)


class CubeCanvas(MyCanvasBase):
    def InitGL(self):
        # set viewing projection
        glMatrixMode(GL_PROJECTION)
        glFrustum(-0.5, 0.5, -0.5, 0.5, 1.0, 3.0)

        # position viewer
        glMatrixMode(GL_MODELVIEW)
        glTranslatef(0.0, 0.0, -2.0)

        # position object
        glRotatef(self.y, 1.0, 0.0, 0.0)
        glRotatef(self.x, 0.0, 1.0, 0.0)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)


    def OnDraw(self):
        # clear color and depth buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # draw six faces of a cube
        glBegin(GL_QUADS)
        glNormal3f( 0.0, 0.0, 1.0)
        glVertex3f( 0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(-0.5,-0.5, 0.5)
        glVertex3f( 0.5,-0.5, 0.5)

        glNormal3f( 0.0, 0.0,-1.0)
        glVertex3f(-0.5,-0.5,-0.5)
        glVertex3f(-0.5, 0.5,-0.5)
        glVertex3f( 0.5, 0.5,-0.5)
        glVertex3f( 0.5,-0.5,-0.5)

        glNormal3f( 0.0, 1.0, 0.0)
        glVertex3f( 0.5, 0.5, 0.5)
        glVertex3f( 0.5, 0.5,-0.5)
        glVertex3f(-0.5, 0.5,-0.5)
        glVertex3f(-0.5, 0.5, 0.5)

        glNormal3f( 0.0,-1.0, 0.0)
        glVertex3f(-0.5,-0.5,-0.5)
        glVertex3f( 0.5,-0.5,-0.5)
        glVertex3f( 0.5,-0.5, 0.5)
        glVertex3f(-0.5,-0.5, 0.5)

        glNormal3f( 1.0, 0.0, 0.0)
        glVertex3f( 0.5, 0.5, 0.5)
        glVertex3f( 0.5,-0.5, 0.5)
        glVertex3f( 0.5,-0.5,-0.5)
        glVertex3f( 0.5, 0.5,-0.5)

        glNormal3f(-1.0, 0.0, 0.0)
        glVertex3f(-0.5,-0.5,-0.5)
        glVertex3f(-0.5,-0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5,-0.5)
        glEnd()

        if self.size is None:
            self.size = self.GetClientSize()
        w, h = self.size
        w = max(w, 1.0)
        h = max(h, 1.0)
        xScale = 180.0 / w
        yScale = 180.0 / h
        glRotatef((self.y - self.lasty) * yScale, 1.0, 0.0, 0.0);
        glRotatef((self.x - self.lastx) * xScale, 0.0, 1.0, 0.0);

        self.SwapBuffers()

if __name__ == "__main__":
    main(sys.argv)
