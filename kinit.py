# ------------------------------------------------------------------------------
# Module to prevent that the connection breaks after 5 hours.
# ------------------------------------------------------------------------------

import pexpect

child = pexpect.spawn('kinit')

child.expect('Password for {}@{}:'.format('aforell', 'AD.IGD.FRAUNHOFER.DE'))

child.sendline('bO6€DbM3#aPR')

child.send('\n')
