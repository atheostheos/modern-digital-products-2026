# WebSocket DAT callbacks shared by all ml_* nodes: forward everything to the parent COMP's extension.

def onConnect(dat):
	parent().OnConnect()

def onDisconnect(dat):
	parent().OnDisconnect()

def onReceiveText(dat, rowIndex, message):
	parent().OnReceiveText(message)

def onReceiveBinary(dat, contents):
	return

def onReceivePing(dat, contents):
	dat.sendPong(contents)

def onReceivePong(dat, contents):
	return

def onMonitorMessage(dat, message):
	return
