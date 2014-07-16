from abaqusGui import *


###########################################################################
# Class definition
###########################################################################

class TestDB(AFXDataDialog):

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, form):

        # Construct the base class.
        #
        AFXDataDialog.__init__(self, form, 'Test Dialog',
            self.OK|self.CANCEL, DECOR_RESIZE|DIALOG_ACTIONS_SEPARATOR)
        
        # Create the contents of the dialog
        #
        va = AFXVerticalAligner(self)
        AFXTextField(va, 10, 'String:', form.kw1, 0)
        AFXTextField(va, 10, 'Integer:', form.kw2, 0)
        AFXTextField(va, 10, 'Float:', form.kw3, 0)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def show(self):
        
        # Note: This method is only necessary because the prototype
        # application allows changes to be made in the dialog code and
        # reloaded while the application is still running. Normally you
        # would not need to have a show() method in your dialog.
        
        # Resize the dialog to its default dimensions to account for
        # any widget changes that may have been made.
        #
        self.resize(self.getDefaultWidth(), self.getDefaultHeight())
        AFXDataDialog.show(self)

