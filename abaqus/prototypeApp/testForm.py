from abaqusGui import *
import testDB
# Note: The above form of the import statement is used for the prototype
# application to allow the module to be reloaded while the application is
# still running. In a non-prototype application you would use the form:
# from myDB import MyDB


###########################################################################
# Class definition
###########################################################################

class TestForm(AFXForm):

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, owner):

        # Construct the base class.
        #
        AFXForm.__init__(self, owner)
                
        # Command
        #
        self.cmd = AFXGuiCommand(self, 'myCommand', 'myObject')
        
        self.kw1 = AFXStringKeyword(self.cmd, 'kw1', TRUE)
        self.kw2 = AFXIntKeyword(self.cmd, 'kw2', TRUE)
        self.kw3 = AFXFloatKeyword(self.cmd, 'kw3', TRUE)
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def getFirstDialog(self):

        # Note: The style below is used for the prototype application to 
        # allow the dialog to be reloaded while the application is
        # still running. In a non-prototype application you would use:
        #
        # return MyDB(self)
        
        # Reload the dialog module so that any changes to the dialog 
        # are updated.
        #
        reload(testDB)
        return testDB.TestDB(self)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def issueCommands(self):
    
        # Since this is a prototype application, just write the command to
        # the Message Area so it can be visually verified. If you have 
        # defined a "real" command, then you can comment out this method to
        # have the command issued to the kernel.
        #
        # In a non-prototype application you normally do not need to write
        # the issueCommands() method.
        #
        cmds = self.getCommandString()
        getAFXApp().getAFXMainWindow().writeToMessageArea(cmds)
        self.deactivateIfNeeded()
        return TRUE
      
